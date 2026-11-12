"""断句策略一：标点断句。

做法：读 token 序列，攒到某个 token 尾部带标点就断一条。不控字数、不合并、
不聚组、不做任何择优 —— 断点完全由识别结果自带的标点决定。

⚠️ 本文件是“按标点”档的全部实现。下面这张标点清单只服务本档；
默认/频繁/沉浸由 split_presets.py 独立维护参数与边界规则。
"""
from __future__ import annotations

from .split_base import cues_from_cuts, finalize
from .tokenize import Token, build_token_stream

# ---- 断点清单：只有出现在这里的标点才会断开 ----
# 句末停顿（一句话结束）
_SENTENCE_END = set("。！？")

# 句内停顿（意群之间）
_CLAUSE_PAUSE = set("，；")

# 半角句读（中文口语里也常夹带）
_HALF_WIDTH = set(".,!?;")

# 有意**不**收录的两类：
#   1) 顿号 、 冒号 ：: 省略号 … —— 列举、引出、话没说完，多数在一句话内部，
#      断在它们后面会把一个完整意思拆碎
#   2) 百分号 %、各种括号与书名号 （）《》「」【】、间隔号 ·、引号 ——
#      虽属 Unicode 标点类，但根本不是停顿（"达到 380%"、"《产品》"）
STOP_CHARS = _SENTENCE_END | _CLAUSE_PAUSE | _HALF_WIDTH


def is_stop(ch: str) -> bool:
    """这个字符算不算"该断的标点"。"""
    return ch in STOP_CHARS


def tail_has_stop(punct: str) -> bool:
    """token 尾部的标点串里，有没有该断的。"""
    return any(is_stop(c) for c in punct)


def split_by_punct(segments: list[dict], language: str) -> list[dict]:
    """标点直断：每个以标点收尾的子句一条字幕。

    token 化那一层已经保证了切点落在词边界上，所以这里即使只看标点，
    也不会把词切开。
    """
    out: list[dict] = []
    tokens = build_token_stream(segments, language)
    cur: list[Token] = []
    for t in tokens:
        cur.append(t)
        if tail_has_stop(t.punct):
            out.extend(cues_from_cuts(cur, [0]))
            cur = []
    if cur:                          # 结尾没有标点，剩下的也收一条
        out.extend(cues_from_cuts(cur, [0]))
    return finalize(out)
