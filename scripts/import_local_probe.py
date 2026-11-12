"""把本地 Fun-ASR-Nano 的验证结果导入资产库，供 UI 直接预览。

做三件事：
1. 读 results/probe/*.json 的本地识别结果（含 CTC 字级时间戳）
2. 过滤纯标点 token 后转成项目约定的 segments 结构，注册成「识别结果」资产，
   并挂到同名音频资产上（这样预览时有音频、点字幕能跳转）
3. 按四个档位各导出一份字幕资产

只新增资产，不改动、不删除任何已有资产。默认 dry-run，加 --write 才落盘。

用法：
    python scripts/import_local_probe.py            # 预览将要导入什么
    python scripts/import_local_probe.py --write    # 真正写入资产库
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import assets as assets_mod, config, export_split, srt  # noqa: E402
from app.tokenize import _align_words, is_punct_char  # noqa: E402

MODEL_TAG = "Fun-ASR-Nano 本地"
SENT_END = "。！？!?…"
TRAIL = "。！？!?…\"'”’）)】》"

PRESETS = [("punct", "按标点"), ("default", "默认"),
           ("frequent", "频繁"), ("immersive", "沉浸")]


def split_sentences(text: str) -> list[tuple[int, int]]:
    """按句末标点切分，返回 [(char_start, char_end), ...]，保留标点。"""
    out: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in SENT_END:
            j = i + 1
            while j < n and text[j] in TRAIL:
                j += 1
            out.append((start, j))
            while j < n and text[j].isspace():
                j += 1
            start = j
            i = j
        else:
            i += 1
    if start < n:
        out.append((start, n))
    return [(s, e) for s, e in out if text[s:e].strip()]


def build_segments(text: str, words: list[dict]) -> list[dict]:
    """按句切分为 segments，每段带自己的 words（words 不含标点）。

    先用生产代码 _align_words 拿到每个词的字符位置与时间，再按句子区间分配，
    避免自己猜词在原文里的位置（重复词会猜错）。
    """
    units = _align_words(text, words)
    if not units:
        return []

    segs: list[dict] = []
    ui = 0
    for cs, ce in split_sentences(text):
        seg_units = []
        while ui < len(units) and units[ui].start < ce:
            seg_units.append(units[ui])
            ui += 1
        if not seg_units:
            continue
        segs.append({
            "begin": seg_units[0].begin,
            "end": seg_units[-1].end,
            "text": text[cs:ce].strip(),
            "words": [{"text": text[u.start:u.stop],
                       "begin": u.begin, "end": u.end} for u in seg_units],
        })
    return segs


def load_words(probe: dict) -> list[dict]:
    """CTC token -> 项目 words 结构；滤掉纯标点（_align_words 会自己吸附）。"""
    out = []
    for t in probe.get("timestamps") or []:
        tok = t.get("token") or ""
        if not tok.strip() or all(is_punct_char(c) for c in tok):
            continue
        out.append({
            "text": tok,
            "begin": int(round(float(t["start_time"]) * 1000)),
            "end": int(round(float(t["end_time"]) * 1000)),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="真正写入资产库")
    args = ap.parse_args()

    probe_files = sorted((BASE_DIR / "results" / "probe").glob("*.json"))
    if not probe_files:
        print("results/probe/ 下没有识别结果，先跑 scripts/asr_local_probe.py")
        return 1

    # 直接构造独立实例：离线写入、无并发，不必拉起 jobs 的线程池
    store = assets_mod.AssetStore(config.ASSET_DIR)
    store.load()

    all_assets = store.list()
    print(f"资产库现有 {len(all_assets)} 项")

    for probe_file in probe_files:
        probe = json.loads(probe_file.read_text(encoding="utf-8"))
        text = probe.get("text") or ""
        words = load_words(probe)
        if not words:
            print(f"[跳过] {probe_file.name} 没有可用 token")
            continue

        segs = build_segments(text, words)
        if not segs:
            print(f"[跳过] {probe_file.name} 切句失败")
            continue

        stem = probe_file.stem
        duration = segs[-1]["end"] / 1000.0

        # 找同名音频资产，建立来源链
        audio = next((a for a in all_assets
                      if a.kind == assets_mod.KIND_AUDIO
                      and Path(a.name).stem == stem), None)

        elapsed = float(probe.get("_elapsed_sec") or 0)
        speed = f"  识别 {elapsed:.1f}s" if elapsed else "  识别耗时未记录"
        print(f"\n素材: {stem}")
        print(f"  句数 {len(segs)}  时长 {duration:.1f}s{speed}  "
              f"音频资产 {'有 ' + audio.id if audio else '无（预览时没有音频）'}")
        print(f"  首句: [{segs[0]['begin']}ms] {segs[0]['text'][:30]}")
        print(f"  末句: [{segs[-1]['end']}ms] {segs[-1]['text'][:30]}")

        dup = next((a for a in all_assets
                    if a.kind == assets_mod.KIND_SEGMENTS
                    and a.model == MODEL_TAG and a.name.startswith(stem)), None)
        if dup:
            print(f"  [已存在] {dup.id}，跳过（如需重导请先在 UI 删除）")
            continue

        if not args.write:
            print("  [dry-run] 未写入")
            continue

        seg_asset = store.add_segments(
            segs, stem, "zh", MODEL_TAG, source_id=audio.id if audio else "",
            duration_sec=duration, elapsed_sec=elapsed)
        print(f"  [已导入] 识别结果 {seg_asset.id}")

        with tempfile.TemporaryDirectory(prefix="import_sub_") as tmp:
            for code, tag in PRESETS:
                stats: dict = {}
                cues = export_split.split_segments_by_preset(segs, code, stats=stats)
                if not cues:
                    continue
                srt_p = Path(tmp) / f"{stem}_{tag}.srt"
                txt_p = Path(tmp) / f"{stem}_{tag}.txt"
                srt_p.write_text(srt.segments_to_srt(cues), encoding="utf-8")
                txt_p.write_text(srt.segments_to_txt(cues), encoding="utf-8")
                sub = store.add_subtitle(
                    srt_p, txt_p, srt_p.name, "zh", MODEL_TAG, code,
                    len(cues), source_id=seg_asset.id, duration_sec=duration)
                print(f"  [已导入] 字幕 {tag} {sub.id} ({len(cues)} 条)")

    if not args.write:
        print("\n以上为预演。确认无误后加 --write 真正写入。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
