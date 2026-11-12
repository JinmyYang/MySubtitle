"""字幕 token 化：把 ASR 的句级结果转成「不可再分的最小切分单元」。

为什么需要这一层（重构前的三类问题）：

1. ASR 词序列与原文可以 O(n) 双指针精确对齐，不需要模糊匹配。
   实测（统一标点口径后）对齐成功率：中文 94%。
2. ASR 的 word 不是语义词。fun-asr 会把 NVIDIA 切成 N+VIDIA、GTC 切成 G+TC、
   what's 切成 what+'+s。所以"切点落在 ASR 词边界"不足以保证不切开词。
3. 字幕条 = token 区间 [i, j)。token 边界同时是时间边界与语义边界，
   于是「不切词 / 文本无损 / 时间精确 / 时间单调」从"事后断言"变成结构保证。

token 的文本一律取原文切片，因此拼接必然等于原文，不可能有损。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import jieba

# 单条字幕最长时长（毫秒）
MAX_DURATION_MS = 6000

# 词内标点：不是断句点，只是词的一部分（撇号、连字符、引号）
_IN_WORD_PUNCT = set("'\"`~-–—")
# ASR 偶尔会把成对符号挂到错误的词上（例如上一句末尾没有“””，
# 下一句开头却带着“”结果”）。下面两组用于把符号重新贴回语义所在的一侧。
_OPEN_EDGE = set("“‘（(【[《<「『")
_CLOSE_EDGE = set("”’）)】]》>」』")

# ---------------- 文本工具 ----------------

def is_punct_char(ch: str) -> bool:
    """这个字符是不是标点符号（撇号/连字符/引号属于词内字符，不算）。

    ⚠️ 只回答"是不是标点"，**不回答"算不算停顿"**。
    后者是各断句策略自己的事，见 split_punct.py 与 split_dp.py 里各自的清单。
    这里只用于分词：把标点从正文里分出来，决定它跟谁待在同一个 token 里。
    """
    if ch in _IN_WORD_PUNCT:
        return False
    return unicodedata.category(ch)[0] == "P"


def _char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def text_width(text: str) -> int:
    """显示宽度：CJK/全角 = 2，其余 = 1。"""
    return sum(_char_width(c) for c in text)


def tail_punct(s: str) -> str:
    """取文本尾部连续的断句标点（先剥掉尾随空白）。"""
    t = s.rstrip()
    i = len(t)
    while i > 0 and is_punct_char(t[i - 1]):
        i -= 1
    return t[i:]


def has_content(s: str) -> bool:
    """除了空白与断句标点之外还有内容吗（用于识别纯标点 token）。"""
    return any(not ch.isspace() and not is_punct_char(ch) for ch in s)


# ---------------- token ----------------

@dataclass
class Token:
    """不可再分的最小切分单元。

    text   —— 原文切片（含词内空白与尾部标点），拼接必然还原原文
    begin  —— 起始时间（毫秒），来自 ASR 首词
    end    —— 结束时间（毫秒），来自 ASR 末词
    punct  —— 尾部断句标点（'' 表示此处不是标点停顿）

    没有词级时间戳时（_fallback_units）begin/end 是按字符比例插值的估算值，
    时间精度不如真实词时间戳，但顺序与区间覆盖仍然正确。
    """
    text: str
    begin: int
    end: int
    punct: str = ""


@dataclass
class _Unit:
    """构造过程中的中间单元（原文区间 + 时间）。"""
    start: int      # 原文起始偏移
    stop: int       # 正文结束偏移（不含尾部断句标点）
    pstop: int      # 含尾部标点后的结束偏移
    begin: int
    end: int


# ---------------- 对齐：原文 <-> ASR 词序列 ----------------

def _align_words(text: str, words: list[dict]) -> list[_Unit] | None:
    """O(n) 双指针精确对齐。

    逐词消费原文：跳过空白 -> 匹配词的非空白字符 -> 吃掉紧随的断句标点。
    任一步不匹配即返回 None（调用方降级），不做任何模糊近似。
    """
    ti, n = 0, len(text)
    units: list[_Unit] = []
    for w in words:
        while ti < n and text[ti].isspace():
            ti += 1
        start = ti
        for c in w["text"]:
            if c.isspace():
                continue
            if ti >= n or text[ti] != c:
                return None
            ti += 1
        stop = ti
        while ti < n and is_punct_char(text[ti]):
            ti += 1
        units.append(_Unit(start, stop, ti, w["begin"], w["end"]))
    while ti < n and text[ti].isspace():
        ti += 1
    return units if ti == n else None


def _fallback_units(text: str, begin: int, end: int,
                    language: str) -> list[_Unit]:
    """无词级时间戳（或对齐失败）时：中文用 jieba，时间按字符比例插值。"""
    span = max(1, end - begin)
    total = max(1, len(text))

    def mk(s: int, e: int) -> _Unit:
        i = e
        while i > s and is_punct_char(text[i - 1]):
            i -= 1
        return _Unit(s, i, e,
                     begin + span * s // total,
                     begin + span * max(i, s + 1) // total)

    units: list[_Unit] = []
    pos = 0
    for w in jieba.cut(text, cut_all=False):
        units.append(mk(pos, pos + len(w)))
        pos += len(w)
    return units


def _jieba_bounds(text: str) -> set[int]:
    """jieba 词边界位置集合（中文判定"这里能不能切"用）。"""
    bs, pos = set(), 0
    for w in jieba.cut(text, cut_all=False):
        pos += len(w)
        bs.add(pos)
    return bs


def _can_cut(prev: _Unit, nxt: _Unit, language: str,
             bounds: set[int] | None, text: str) -> bool:
    """两个相邻单元之间是否允许切分。

    规则：
      中文 —— 拉丁字母被切碎时合并（N+VIDIA）；其余交给 jieba 词边界
      通用 —— 前一单元以断句标点收尾时必然可切
    """
    if prev.pstop > prev.stop:          # 标点之后是天然断句点
        return True
    if not text[nxt.start:nxt.pstop].strip():
        return False                    # 纯空白块并入前块（否则切片会漏掉它）
    a = text[prev.stop - 1] if prev.stop > prev.start else ""
    b = text[nxt.start] if nxt.start < len(text) else ""
    if not a or not b:
        return True
    if a.isascii() and a.isalnum() and b.isascii() and b.isalnum():
        return False                     # N|VIDIA、G|TC
    if bounds is None:
        return True
    return prev.stop in bounds


def build_tokens(seg: dict, language: str) -> list[Token]:
    """把一句 ASR 结果构造成 token 序列。"""
    text = seg["text"]
    n = len(text)
    if not text.strip():
        return []

    words = [w for w in (seg.get("words") or []) if (w.get("text") or "").strip()]
    units = _align_words(text, words) if words else None
    if not units:
        units = _fallback_units(text, seg["begin"], seg["end"], language)
    if not units:
        return []

    # 合并不可切的相邻单元
    bounds = _jieba_bounds(text)
    merged: list[_Unit] = []
    for u in units:
        if merged and not _can_cut(merged[-1], u, language, bounds, text):
            p = merged[-1]
            p.stop, p.pstop, p.end = u.stop, u.pstop, u.end
        else:
            merged.append(u)

    # 按原文区间切片：文本绝对无损（空白与标点都在切片里）
    tokens: list[Token] = []
    head = ""       # 首个有内容的 token 之前的部分（前导空白），不能丢
    for i, u in enumerate(merged):
        stop = merged[i + 1].start if i + 1 < len(merged) else n
        raw = text[u.start:stop]
        if not raw:
            continue
        # 纯空白/纯标点块不独立成条，并入上一条末尾（保证文本无损、标点不单行）
        if not has_content(raw):
            if tokens:
                tokens[-1].text += raw
                tokens[-1].punct = tail_punct(tokens[-1].text)
            else:
                head += raw
            continue
        tk = Token(text=raw, begin=u.begin, end=u.end,
                   punct=tail_punct(raw))
        if head:
            tk.text = head + tk.text
            head = ""
        tokens.append(tk)

    return normalize_token_stream(tokens)


def normalize_token_stream(tokens: list[Token]) -> list[Token]:
    """跨多个 ASR 句做一次 token 流级规整。

    `build_tokens` 只看单句，闭引号如果落在下一句开头，它没有上一句可贴。
    这里把多句拼成流后再回收一次成对符号，让标点与词都不断在 ASR 句界上。
    """
    _rebalance_pair_punct(tokens)
    _clear_numeric_punct(tokens)
    return tokens


def build_token_stream(segments: list[dict], language: str) -> list[Token]:
    """把整段 ASR 结果拼成一条 token 流，并做跨句符号规整。"""
    tokens: list[Token] = []
    for seg in segments:
        tokens.extend(build_tokens(seg, language))
    return normalize_token_stream(tokens)


def _rebalance_pair_punct(tokens: list[Token]) -> None:
    """把成对符号移回语义正确的一侧，文本仍保持原样拼接。

    ASR 的 word 时间戳按音频切，不保证包住标点：有时“””落在下一句开头，
    有时““”黏在上一个词尾。这里做一次纯文本搬移：
      - 行首的闭符号（”）》）贴回前一个 token；
      - 行尾的开符号（“（《）交给后一个 token。
    这样断句时不会出现“上一行以‘“’结尾、下一行以‘”’开头”的割裂。
    """
    for i in range(1, len(tokens)):
        raw = tokens[i].text
        lead = len(raw) - len(raw.lstrip())
        body = raw[lead:]
        k = 0
        while k < len(body) and body[k] in _CLOSE_EDGE:
            k += 1
        if k:
            tokens[i - 1].text += raw[:lead + k]
            tokens[i - 1].punct = tail_punct(tokens[i - 1].text)
            tokens[i].text = body[k:]
            tokens[i].punct = tail_punct(tokens[i].text)

    for i in range(len(tokens) - 1):
        t = tokens[i].text.rstrip()
        k = len(t)
        while k > 0 and t[k - 1] in _OPEN_EDGE:
            k -= 1
        if k < len(t):
            moved = t[k:]
            tail_ws = tokens[i].text[len(t):]
            tokens[i].text = t[:k] + tail_ws
            tokens[i].punct = tail_punct(tokens[i].text)
            tokens[i + 1].text = moved + tokens[i + 1].text
            tokens[i + 1].punct = tail_punct(tokens[i + 1].text)


def _clear_numeric_punct(tokens: list[Token]) -> None:
    """把"数字里的标点"从断点标记里摘掉 —— 小数点、千分位逗号。

    半角的 `.` `,` 也是断句标点，但出现在数字里时是另一回事：
    `26.6` / `3.8%` / `1.14` / `650,000`。不分清就会把意群切在数字中间，
    典型错误："场均还有1." | "6个抢断"。

    判定：标点的**左右都是数字**才算数字内部标点（`U.S.`、`etc.` 不受影响）。
    """
    for i in range(len(tokens) - 1):
        tk, nxt = tokens[i], tokens[i + 1]
        p = tk.punct
        if not p or p.strip(".,") or ("." not in p and "," not in p):
            continue                      # 尾部标点不只由 . / , 组成，跳过
        body = tk.text.rstrip().rstrip(".,").rstrip()
        if body[-1:].isdigit() and nxt.text.lstrip()[:1].isdigit():
            tk.punct = ""                 # 标点仍留在正文里，只是不再标记为可断处
