"""本地 Fun-ASR-Nano 时间戳探针。

目的：验证 ModelScope 版 checkpoint（README 称已含 86 个 CTC 张量）能否输出
原生字级时间戳，并把真实返回结构打印出来，供后续映射到 app 的
segments[].words[] 参考。

不改 app/ 下任何文件；模型与输出都不入 git（.runtime/ results/ 已在 .gitignore）。

用法：
    python scripts/asr_local_probe.py --download-only   # 只拉模型
    python scripts/asr_local_probe.py                   # 默认挑最短素材跑通
    python scripts/asr_local_probe.py --audio "测试音频/xxx.m4a"
    python scripts/asr_local_probe.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
MODEL_DIR = BASE_DIR / ".runtime" / "models" / "Fun-ASR-Nano-2512"
# remote code：ModelScope 模型仓库不含 model.py，需从 GitHub 仓库取。
# model.py 内是 `from ctc import CTC` / `from tools.utils import forced_align`
# 这类绝对导入，所以必须把仓根放进 sys.path。
CODE_DIR = BASE_DIR / ".runtime" / "fun_asr_repo"
OUT_DIR = BASE_DIR / "results" / "probe"


def _json_safe(obj):
    """结果里可能混入 numpy 标量，统一兜底成字符串。"""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def download_model() -> Path:
    from modelscope import snapshot_download

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[下载] {MODEL_ID}")
    print(f"[目标] {MODEL_DIR}")
    print("[提示] 约 2GB，首次会走 ModelScope，断点可续传")
    t0 = time.time()
    path = snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR))
    print(f"[下载] 完成，耗时 {time.time() - t0:.1f}s: {path}")
    return Path(path)


def pick_audio(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            raise SystemExit(f"音频不存在: {p}")
        return p
    cands = sorted((BASE_DIR / "测试音频").glob("*.m4a"), key=lambda p: p.stat().st_size)
    if not cands:
        raise SystemExit("测试音频/ 下没有 m4a 素材")
    return cands[0]


def report(item: dict) -> None:
    print("\n" + "=" * 60)
    print("顶层字段")
    print("=" * 60)
    for k, v in item.items():
        if isinstance(v, (list, tuple)):
            print(f"  {k:18s} {type(v).__name__:6s} len={len(v)}")
        elif isinstance(v, str):
            print(f"  {k:18s} str    len={len(v)}")
        elif isinstance(v, dict):
            print(f"  {k:18s} dict   keys={list(v)[:6]}")
        else:
            print(f"  {k:18s} {type(v).__name__} = {v}")

    text = item.get("text") or ""
    print("\n" + "=" * 60)
    print(f"识别文本（共 {len(text)} 字，显示前 200）")
    print("=" * 60)
    print(text[:200])

    ctc_text = (item.get("ctc_text") or "").replace("<|nospeech|>", "")
    if ctc_text:
        print("\n" + "=" * 60)
        flag = "一致" if ctc_text == text else "不一致"
        print(f"CTC 纯声学文本（共 {len(ctc_text)} 字，与上文本{flag}）")
        print("=" * 60)
        print(ctc_text[:200])

    for field in ("timestamp", "timestamps"):
        v = item.get(field)
        if not v:
            print(f"\n[空] {field} 没有内容")
            continue
        print("\n" + "=" * 60)
        print(f"{field}（共 {len(v)} 项，前 8 项）")
        print("=" * 60)
        print(f"  元素类型: {type(v[0]).__name__}")
        if isinstance(v[0], dict):
            print(f"  元素字段: {list(v[0].keys())}")
        print("  " + json.dumps(v[:8], ensure_ascii=False, default=_json_safe))

        # 粗略判断单位：末尾时间若远大于音频秒数，说明是毫秒
        last = v[-1]
        tail = last.get("end_time") if isinstance(last, dict) else (
            last[-1] if isinstance(last, (list, tuple)) else None
        )
        if isinstance(tail, (int, float)):
            print(f"  末项结束值 = {tail}（若数值接近音频总秒数则为秒，若大 1000 倍则为毫秒）")

    si = item.get("sentence_info")
    if si:
        print("\n" + "=" * 60)
        print(f"sentence_info（共 {len(si)} 句，前 2 句）")
        print("=" * 60)
        for s in si[:2]:
            print(json.dumps(s, ensure_ascii=False, default=_json_safe)[:700])
            print("  ---")
    else:
        print("\n[空] sentence_info 没有内容（可能没配 punc_model 或未开 sentence_timestamp）")


def check_alignment(item: dict) -> None:
    """用生产代码的 _align_words 检验 CTC token 能否对上识别文本。

    这一步决定成败：对不上时 tokenize 会静默降级到 jieba + 比例插值，
    断句仍能出结果，但时间轴精度全丢，且不会报错。
    """
    sys.path.insert(0, str(BASE_DIR))
    from app.tokenize import _align_words, is_punct_char

    text = item.get("text") or ""
    ts = item.get("timestamps") or []
    print("\n" + "=" * 60)
    print("对齐检查（直接调用 app.tokenize._align_words）")
    print("=" * 60)
    if not text or not ts:
        print("  [跳过] 缺少 text 或 timestamps")
        return

    words = []
    for t in ts:
        if not isinstance(t, dict):
            continue
        tok = t.get("token") or t.get("text") or ""
        if not tok.strip():
            continue
        st = float(t.get("start_time") or 0)
        en = float(t.get("end_time") or 0)
        words.append({"text": tok, "begin": int(st * 1000), "end": int(en * 1000)})

    joined = "".join(w["text"] for w in words)
    print(f"  text 长度        : {len(text)}")
    print(f"  token 数         : {len(words)}")
    print(f"  token 拼接长度   : {len(joined)}")
    print(f"  token 拼接前 100 : {joined[:100]}")

    # 直接对齐：CTC 的 token 序列自带标点，而 _align_words 消费完一个字后
    # 会自动跳过原文标点，两者叠加导致标点被消费两次 → 必然失败。
    print(f"  原始 token {len(words)} 个：")
    units_raw = _align_words(text, words)
    print(f"    直接对齐 -> {'成功' if units_raw else '失败（标点被重复消费）'}")

    # 过滤掉纯标点 token 后，交给 _align_words 自己吸附标点 —— 这才是正确用法。
    kept = [w for w in words if not all(is_punct_char(c) for c in w["text"])]
    print(f"  过滤纯标点后 {len(kept)} 个：")
    units = _align_words(text, kept)
    if units:
        bad = sum(1 for a, b in zip(units, units[1:]) if b.begin < a.begin)
        print(f"    对齐 -> 成功，单元 {len(units)} 个")
        print(f"      首 {units[0].begin}ms，末 {units[-1].end}ms，时间倒序处数 {bad}")
        print("      => 可直接接入现有断句引擎，无需改动 tokenize.py")
        return

    print("    对齐 -> 失败，会降级到 jieba + 比例插值")
    ti = 0
    for idx, w in enumerate(kept):
        while ti < len(text) and text[ti].isspace():
            ti += 1
        for c in w["text"]:
            if c.isspace():
                continue
            if ti >= len(text) or text[ti] != c:
                got = text[ti] if ti < len(text) else "<越界>"
                print(f"  首个不匹配: token#{idx} {w['text']!r}")
                print(f"    原文位置 {ti} 期望 {w['text'][:1]!r} 实际 {got!r}")
                print(f"    原文此处: {text[max(0, ti - 15):ti + 25]!r}")
                return
            ti += 1
        while ti < len(text) and is_punct_char(text[ti]):
            ti += 1
    print(f"  消费到 {ti}/{len(text)}，尾部残留: {text[ti:ti + 40]!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="指定音频，默认取最短素材")
    ap.add_argument("--download-only", action="store_true", help="只下载模型")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu，默认自动")
    args = ap.parse_args()

    if not (MODEL_DIR / "model.pt").exists():
        download_model()
    else:
        size_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / 1024**3
        print(f"[模型] 已存在: {MODEL_DIR}（{size_gb:.2f} GB）")

    if args.download_only:
        return 0

    import torch
    from funasr import AutoModel

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    if device.startswith("cuda"):
        print(f"[显存] 空闲/总量: {torch.cuda.mem_get_info()}")

    kwargs = dict(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model="ct-punc",
        device=device,
        disable_update=True,
    )
    remote_code = CODE_DIR / "model.py"
    if remote_code.exists():
        if str(CODE_DIR) not in sys.path:
            sys.path.insert(0, str(CODE_DIR))
        kwargs["remote_code"] = str(remote_code)
        print(f"[远程代码] {remote_code}")
    else:
        print(f"[警告] 未找到 {remote_code}，尝试交给 funasr 自行解析")

    t0 = time.time()
    model = AutoModel(**kwargs)
    print(f"[加载] 耗时 {time.time() - t0:.1f}s")

    audio = pick_audio(args.audio)
    probe = audio.stat().st_size
    print(f"[音频] {audio.name}（{probe / 1024**2:.1f} MB）")

    t1 = time.time()
    res = model.generate(
        input=[str(audio)],
        cache={},
        batch_size=1,
        language="中文",
        itn=True,             # 数字/日期规整，与云端 Fun-ASR 输出风格保持一致
        sentence_timestamp=True,
    )
    cost = time.time() - t1
    print(f"[推理] 耗时 {cost:.1f}s")

    if not res:
        print("[失败] generate 返回空")
        return 2
    if not res[0]:
        print("[失败] res[0] 为空，通常是 VAD 没检到语音")
        return 2

    item = res[0]
    report(item)
    check_alignment(item)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{audio.stem}.json"
    out.write_text(
        json.dumps(item, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    print(f"\n[保存] {out}")

    has_token_ts = bool(item.get("timestamp") or item.get("timestamps"))
    print("\n" + "=" * 60)
    print(f"结论：字级时间戳 {'有' if has_token_ts else '没有'}")
    if not has_token_ts:
        print("  → CTC 未生效。检查是否用了 ModelScope 权重（hub=ms）且 funasr >= 1.3.26")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
