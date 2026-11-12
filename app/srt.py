"""SRT 字幕生成与解析工具。"""
from __future__ import annotations

import re
from pathlib import Path


def _ts(ms: int) -> str:
    """毫秒 -> SRT 时间戳 HH:MM:SS,mmm。"""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """[{begin, end, text}] -> SRT 文本。"""
    lines: list[str] = []
    for idx, seg in enumerate(segments, 1):
        lines.append(str(idx))
        lines.append(f"{_ts(seg['begin'])} --> {_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def segments_to_txt(segments: list[dict]) -> str:
    """[{begin, end, text}] -> 纯文本（每句一行）。"""
    return "\n".join(seg["text"] for seg in segments)


_TS_RE = re.compile(
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)


def parse_srt(path: str | Path) -> list[dict]:
    """解析 SRT 文件为片段列表，宽容 BOM、\r\n、空行差异。"""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").strip())
    segments: list[dict] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        ts_line_idx = None
        for i, ln in enumerate(lines):
            if _TS_RE.search(ln):
                ts_line_idx = i
                break
        if ts_line_idx is None:
            continue
        m = _TS_RE.search(lines[ts_line_idx])
        g = [int(x) for x in m.groups()]
        begin = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        end = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        text = " ".join(lines[ts_line_idx + 1:]).strip()
        if not text:
            continue
        segments.append({"begin": begin, "end": end, "text": text})
    segments.sort(key=lambda s: s["begin"])
    return segments
