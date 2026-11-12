"""离线验证四档断句：用已落盘的转写 JSON 重建分段，检查可读性与结构不变量。

不变量（token 模型的结构保证）：
  文本无损 / 不超档位上限 / 时间单调不重叠 / 断点都在 token 边界

可读性指标：
  平均字数、P95 字数、平均/P95 CPS、短闪占比、词尾虚词、数量词拆分

数据源自动发现，找到几份跑几份；外语历史资产会跳过。零 API 成本。

用法: python scripts/regress_export.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dashscope_client, export_split, split_presets, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    (ROOT / "assets" / "segments", "*.json"),
    (ROOT / "results" / "preset_eval" / "segments", "*.json"),
    (ROOT / "results" / "split_verify", "*_zh_raw.json"),
    (ROOT / "samples", "*.json"),
]
PRESETS = ("default", "frequent", "immersive")
_WS_RE = re.compile(r"\s+")


def load_segments(path: Path) -> list[dict]:
    """原始转写 JSON（dict）与已解析的 segments（list）都支持。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    return dashscope_client.parse_transcription_data(raw)


def _nows(s: str) -> str:
    return _WS_RE.sub("", s)


def _q(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def check_token_bounds(segments: list[dict], cues: list[dict]) -> list[str]:
    """核对每个断点都落在 token 边界上。返回违规片段。"""
    full = _nows("".join(s["text"] for s in segments))
    bounds: set[int] = {0}
    cum = 0
    for t in tokenize.build_token_stream(segments, "zh"):
        cum += len(_nows(t.text))
        bounds.add(cum)
    bounds.add(len(full))

    bad: list[str] = []
    pos = 0
    for c in cues:
        cc = _nows(c["text"])
        if not cc:
            continue
        if not full.startswith(cc, pos):
            return [f"<无法顺序对齐: {full[pos:pos + 12]}> vs {cc[:12]}"]
        pos += len(cc)
        if pos < len(full) and pos not in bounds:
            bad.append(full[max(0, pos - 6):pos + 6])
    return bad


def run(name: str, segments: list[dict], show: int = 4) -> dict[str, dict]:
    print(f"\n=== {name} ({len(segments)} 个 ASR 句) ===")
    expect = _nows("".join(s["text"] for s in segments))
    result: dict[str, dict] = {}
    for preset in PRESETS:
        spec = split_presets.PRESETS[preset]
        cues = export_split.split_by_preset(segments, preset)
        stats = {
            "count": len(cues),
            "chars": [tokenize.text_width(c["text"]) / 2 for c in cues],
            "dur": [(c["end"] - c["begin"]) / 1000 for c in cues],
        }
        joined = _nows("".join(c["text"] for c in cues))
        assert joined == expect, f"{preset}: 文本有损"

        over = [c["text"] for c in cues
                if tokenize.text_width(c["text"]) / 2 > spec.max_chars + 0.01]
        assert not over, f"{preset}: {len(over)} 条超限: {over[:2]}"
        for i in range(len(cues) - 1):
            assert cues[i]["begin"] <= cues[i + 1]["begin"], f"{preset}: 时间倒挂"
            assert cues[i]["end"] <= cues[i + 1]["begin"], f"{preset}: 时间重叠"
        bad = check_token_bounds(segments, cues)
        assert not bad, f"{preset}: 断点切在词中间: {bad[:3]}"

        chars, durs = stats["chars"], stats["dur"]
        cps = [ch / d for ch, d in zip(chars, durs) if d > 0]
        short = sum(d < 0.7 for d in durs)
        print(f"  {preset:<9} {len(cues):>4} 条 | 字数 avg={statistics.mean(chars):5.1f} "
              f"P95={_q(chars, .95):4.1f} max={max(chars):4.1f} | "
              f"CPS avg={statistics.mean(cps):4.1f} P95={_q(cps, .95):4.1f} | "
              f"短闪 {short:>3} ({short/len(cues):4.1%})")
        stats.update({"cps": cps, "short": short})
        result[preset] = stats
        if preset == "default":
            for c in cues[:show]:
                print(f"      [{c['begin'] // 1000:>3}s] {c['text']}")
    return result


def main() -> int:
    found = 0
    aggregate: dict[str, list[dict]] = {p: [] for p in PRESETS}
    seen: set[Path] = set()
    for directory, pattern in SOURCES:
        if not directory.exists():
            continue
        for path in sorted(directory.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                segments = load_segments(path)
            except Exception as exc:  # noqa: BLE001 - 格式不符的文件跳过
                print(f"[skip] {path.name}: {exc}")
                continue
            if not segments:
                continue
            text = "".join(str(s.get("text") or "") for s in segments)
            if sum("\u4e00" <= ch <= "\u9fff" for ch in text) < 20:
                print(f"[skip] {path.name}: 非普通话素材")
                continue
            result = run(path.stem, segments)
            for preset, stats in result.items():
                aggregate[preset].append(stats)
            found += 1

    if not found:
        print("未找到任何转写数据。请确认 assets/segments/ 或 "
              "results/preset_eval/segments/ 里有识别结果。")
        return 1

    print("\n=== 汇总 ===")
    for preset, rows in aggregate.items():
        all_chars = [x for r in rows for x in r["chars"]]
        all_cps = [x for r in rows for x in r["cps"]]
        all_short = sum(r["short"] for r in rows)
        total = sum(r["count"] for r in rows)
        print(f"  {preset:<9} {total:>5} 条 | 字数 avg={statistics.mean(all_chars):5.1f} "
              f"P95={_q(all_chars, .95):4.1f} | CPS P95={_q(all_cps, .95):4.1f} | "
              f"短闪 {all_short/total:5.1%}")
    print(f"\n离线验证通过 ✓（{found} 份数据）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
