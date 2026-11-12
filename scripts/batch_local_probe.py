"""批量跑本地 Fun-ASR-Nano，记录每份素材的识别耗时。

模型只加载一次，逐份推理。结果写到 results/probe/<素材名>.json，
结构沿用 asr_local_probe.py，额外加两个字段：
  _elapsed_sec  识别耗时（秒）
  _audio_sec    音频时长（秒）

已存在 probe 结果的默认跳过，加 --force 重跑。

用法：
    python scripts/batch_local_probe.py                # 跳过已有结果
    python scripts/batch_local_probe.py --force        # 全部重跑
    python scripts/batch_local_probe.py --only "厨师长"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / ".runtime" / "models" / "Fun-ASR-Nano-2512"
CODE_DIR = BASE_DIR / ".runtime" / "fun_asr_repo"
OUT_DIR = BASE_DIR / "results" / "probe"
AUDIO_DIR = BASE_DIR / "测试音频"


def audio_seconds(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    ap.add_argument("--only", default="", help="只跑文件名包含该子串的素材")
    args = ap.parse_args()

    files = sorted(AUDIO_DIR.glob("*.m4a"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    if not files:
        print("测试音频/ 下没有匹配的素材")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for f in files:
        done = OUT_DIR / f"{f.stem}.json"
        if done.exists() and not args.force:
            print(f"[跳过] {f.name}（已有结果）")
            continue
        todo.append(f)
    if not todo:
        print("没有需要跑的素材")
        return 0

    import torch
    from funasr import AutoModel

    if not (MODEL_DIR / "model.pt").exists():
        print(f"模型不存在：{MODEL_DIR}")
        return 1

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")
    print(f"[待跑] {len(todo)} 份")

    kwargs = dict(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model="ct-punc",
        device=device,
        disable_update=True,
    )
    if (CODE_DIR / "model.py").exists():
        if str(CODE_DIR) not in sys.path:
            sys.path.insert(0, str(CODE_DIR))
        kwargs["remote_code"] = str(CODE_DIR / "model.py")

    t0 = time.time()
    model = AutoModel(**kwargs)
    print(f"[加载] 耗时 {time.time() - t0:.1f}s\n", flush=True)

    total_audio = 0.0
    total_cost = 0.0
    for i, audio in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {audio.name}", flush=True)
        dur = audio_seconds(audio)
        t1 = time.time()
        try:
            res = model.generate(
                input=[str(audio)], cache={}, batch_size=1,
                language="中文", itn=True, sentence_timestamp=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  失败: {exc}\n", flush=True)
            continue
        cost = time.time() - t1
        if not res or not res[0]:
            print("  结果为空\n", flush=True)
            continue

        item = res[0]
        item["_elapsed_sec"] = round(cost, 2)
        item["_audio_sec"] = round(dur, 2)
        (OUT_DIR / f"{audio.stem}.json").write_text(
            json.dumps(item, ensure_ascii=False, default=str), encoding="utf-8")

        n_ts = len(item.get("timestamps") or [])
        total_audio += dur
        total_cost += cost
        rtf = cost / dur if dur else 0
        print(f"  音频 {dur:.1f}s  识别 {cost:.1f}s  RTF {rtf:.3f}  "
              f"字数 {len(item.get('text') or '')}  token {n_ts}\n", flush=True)

    print("=" * 50)
    print(f"合计音频 {total_audio:.1f}s，识别 {total_cost:.1f}s", end="")
    if total_audio:
        print(f"，平均 RTF {total_cost / total_audio:.3f}")
    else:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
