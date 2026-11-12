"""字幕条组装（所有断句档位共用的地基）。

这一层只做机械转换：把「token 切分结果」变成「带时间戳的字幕条」，
不含任何断句判断、不认识任何标点、不参与任何择优。
"""
from __future__ import annotations

from .tokenize import Token


def cues_from_cuts(tokens: list[Token], cuts: list[int]) -> list[dict]:
    """按切点把 token 序列组装成字幕条。

    cuts 是切点索引列表（含 0），形如 [0, 5, 11] 表示切成 [0,5) [5,11) 两段。
    文本取 token 原样拼接（token 本身是原文切片，所以拼接必然等于原文），
    时间取首 token 的开始与末 token 的结束。
    """
    n = len(tokens)
    out: list[dict] = []
    for a, b in zip(cuts, cuts[1:] + [n]):
        if b <= a:
            continue
        text = "".join(t.text for t in tokens[a:b]).strip()
        if not text:
            continue
        out.append({"text": text, "begin": tokens[a].begin, "end": tokens[b - 1].end})
    return out


def finalize(out: list[dict], min_duration_ms: int = 20) -> list[dict]:
    """补足最短可见时长，再收敛相邻条重叠。

    `min_duration_ms` 是“尽量达到”的显示时长：只有后面的空隙够长时才延长，
    绝不为了凑时长顶过下一条字幕。
    """
    for i, cue in enumerate(out):
        want = max(20, int(min_duration_ms))
        if cue["end"] - cue["begin"] < want:
            cue["end"] = cue["begin"] + want
        if i + 1 < len(out) and cue["end"] > out[i + 1]["begin"]:
            cue["end"] = max(out[i + 1]["begin"], cue["begin"] + 1)
    return out
