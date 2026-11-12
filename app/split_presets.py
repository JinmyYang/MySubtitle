"""字幕断句预设：一套确定性排版引擎，三种节奏档位。

设计不是从“某段素材看起来如何”倒推规则，而是先确定观众读字幕时的硬约束：

1. **看得完**：单条时长、每秒字数（CPS）有上限；太短会闪，太长会滞留。
2. **看得清**：单条宽度受屏幕与两行上限约束，档位只改变目标宽度。
3. **读得顺**：切点只能落在词边界；不跨句末标点；不把“的/地/得”和它前面的成分切开。
4. **不孤立**：尽量不产生一两个字、纯语气词的独立字幕。

`default / frequent / immersive` 共用同一套动态规划，只换目标字数、时长和阅读速度：
- 默认：15 字左右，兼顾完整与节奏；
- 频繁：10 字左右，适合竖屏短视频的快速切换；
- 沉浸：20 字左右，减少跳字，适合访谈、播客、知识内容。

`punct` 不走这套引擎：它保留“按 ASR 标点直断”的语义，用户明确要求逐小句时使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jieba.posseg

from .split_base import finalize
from .split_punct import split_by_punct
from .tokenize import Token, build_tokens, normalize_token_stream, text_width

_INF = float("inf")

# 句末标点：一句话结束，字幕不跨过它
_SENTENCE_END = set("。！？!?…")
# 句内停顿：适合作为字幕落点，但不是硬边界。
# 顿号/冒号只算弱停顿，鼓励在列举项内部收束，但不跨越句末。
_CLAUSE_PAUSE = set("，；、：,;:")
# 超过这个静音间隔，视为不可跨越的语义断点
_HARD_GAP_MS = 700
# 单次 DP 的 token 上限，防止长独白退化成 O(n²)
_MAX_TOKENS_PER_DP = 600


@dataclass(frozen=True)
class Preset:
    """一个档位的排版目标。字数按汉字计，显示宽度另行换算。"""

    code: str
    name: str
    desc: str
    target_chars: int
    max_chars: int
    min_chars: int
    min_duration_ms: int
    max_duration_ms: int
    max_cps: float
    # 落点奖励：句内标点 / ASR 句段边界
    clause_reward: float
    segment_reward: float
    # 非标点处硬切的基准代价（越大越“舍不得”切碎）
    cut_penalty: float
    cross_segment_penalty: float
    # 是否保护“本来就放得下的意群”：默认/沉浸绝不在意群内部切
    protect_clause: bool
    # 切在虚词上的代价
    bad_end_penalty: float
    bad_start_penalty: float


PRESETS: dict[str, Preset] = {
    "default": Preset(
        code="default", name="默认", desc="大多数视频，完整与节奏平衡",
        target_chars=15, max_chars=24, min_chars=4,
        min_duration_ms=900, max_duration_ms=6000, max_cps=9.0,
        clause_reward=70, segment_reward=30,
        cut_penalty=500, cross_segment_penalty=80, protect_clause=True,
        bad_end_penalty=520, bad_start_penalty=180,
    ),
    "frequent": Preset(
        code="frequent", name="频繁", desc="短视频，更短、更快、更跟手",
        target_chars=10, max_chars=15, min_chars=3,
        min_duration_ms=750, max_duration_ms=4500, max_cps=10.0,
        clause_reward=55, segment_reward=20,
        cut_penalty=200, cross_segment_penalty=55, protect_clause=False,
        bad_end_penalty=460, bad_start_penalty=150,
    ),
    "immersive": Preset(
        code="immersive", name="沉浸", desc="内容向，少跳字，保留完整意群",
        target_chars=20, max_chars=30, min_chars=7,
        min_duration_ms=1300, max_duration_ms=7000, max_cps=9.0,
        clause_reward=90, segment_reward=35,
        cut_penalty=620, cross_segment_penalty=110, protect_clause=True,
        bad_end_penalty=620, bad_start_penalty=220,
    ),
}

DEFAULT_PRESET = "default"

# 只对“单字/单个词”判定，避免把“和平”“对于”这类正常词误伤
_BAD_END = set("的得地和与及或而但并在把被将给对向往从跟比就才都也还又再很太更最不没别请让使叫会能要想可应需该")
_BAD_START = set("吗呢吧")
# 多字连接词收尾会把下一句悬空：`……结果 | 不能丢`
_BAD_END_WORDS = {"结果", "但是", "然后", "因为", "所以", "如果", "虽然",
                  "而且", "不过", "于是", "就是", "其实", "以及", "并且",
                  "就像", "比如", "例如", "譬如", "像是"}
# 放在后一句开头会把前一句的修饰关系拦腰截断：`梗图 | 一样的`
_BAD_START_WORDS = ("一样", "似的", "一般地", "相同", "类似")
# 这些字属于前一个词（说|过、他|的），不能出现在行首
_ATTACH_LEFT = set("了着过")
# 成对符号不能拆在开口/闭口处：上一行以“（开头、下一行以”）开头都很怪
_OPEN_TAIL = set("“‘（(【[《<「『")
_CLOSE_HEAD = set("”’）)】]》>」』")
_QUOTE_OPEN = set("“‘「『")
_QUOTE_CLOSE = set("”’」』")
# 数字后面的量词/单位：不能把“10 颗”“3 年”拆到两条
_MEASURE_START = set("颗个位名种次项块张台款年月日岁倍分秒米克吨度页条只件套层期届步")
# 量词收尾会把中心语甩到下一行：`两颗 | 芯粒`、`第一个 | 人`
_CLASSIFIER_END = set("颗个位名种项块张台款条只件套层期届步些")


@dataclass
class _Piece:
    """连续 token 流上的一个 token，带边界语义。"""

    token: Token
    segment_end: bool = False


def _has_sentence_end(text: str) -> bool:
    return any(ch in _SENTENCE_END for ch in text)


def _has_clause_end(text: str) -> bool:
    return any(ch in _CLAUSE_PAUSE for ch in text)


def _core(text: str) -> str:
    """去掉空白与尾部标点，用于虚词判断。"""
    t = text.strip()
    while t and (t[-1].isspace() or t[-1] in _SENTENCE_END | _CLAUSE_PAUSE):
        t = t[:-1]
    return t


@lru_cache(maxsize=8192)
def _pos(text: str) -> tuple[tuple[str, str], ...]:
    return tuple((w.word, w.flag) for w in jieba.posseg.cut(text))


def _is_struct_particle(text: str) -> bool:
    """是否是结构助词“的/地/得”（用词性确认，避免“得分”误判）。"""
    core = _core(text)
    if not core or core[0] not in "的地得":
        return False
    words = _pos(core)
    return bool(words) and words[0][1].startswith("u")


def _ends_with_struct_particle(text: str) -> bool:
    core = _core(text)
    if not core:
        return False
    words = _pos(core)
    return bool(words) and words[-1][0] in "的地得" and words[-1][1].startswith("u")


def _ends_with_bad_function(text: str) -> bool:
    """行尾是不是虚词，即使它被 ASR 并进了更长的词块。

    例：`所有人都` 的末词性为副词“都”，同样不能收尾；
    但 `目的` 的末词是名词，不能误伤。
    """
    core = _core(text)
    if not core:
        return False
    if len(core) == 1 and core in _BAD_END:
        return True
    words = _pos(core)
    if not words:
        return False
    word, flag = words[-1]
    return len(word) == 1 and word in _BAD_END and flag[:1] in "dcpu"


def _starts_with_left_particle(text: str) -> bool:
    """行首是不是“了/着/过”这种应黏在前词后的助词。"""
    core = _core(text)
    if not core or core[0] not in _ATTACH_LEFT:
        return False
    words = _pos(core)
    return bool(words) and words[0][0] in _ATTACH_LEFT and (
        words[0][1].startswith("u") or len(words) > 1)


def _verb_object_boundary(prev: Token, nxt: Token) -> bool:
    """动词和宾语之间不是理想断点（`选择 | 我们这种蒜`）。

    加中等代价而不是硬禁：长句没有标点时仍然允许切开，但优先在逗号/句段边界收束。
    带“了/着/过”的动词短语已经完整，不算。
    """
    a, b = _core(prev.text), _core(nxt.text)
    if not a or not b or a[-1] in "了着过":
        return False
    pa, pb = _pos(a), _pos(b)
    if not pa or not pb:
        return False
    return pa[-1][1].startswith("v") and pb[0][1] in {"n", "ns", "nt", "nz", "nr", "r", "rz"}


def _bad_lexical_boundary(prev: Token, nxt: Token) -> bool:
    """数量词与型号中间不该拆开：`10 | 颗`、`3 | 年`、`叉 | 925`。"""
    a, b = _core(prev.text), _core(nxt.text)
    if not a or not b:
        return False
    if a[-1].isdigit() and b[0] in _MEASURE_START:
        return True
    # 原文里紧贴的中文+数字（N950、叉925）通常是同一个型号，不要拆
    no_space = not prev.text[-1:].isspace() and not nxt.text[:1].isspace()
    if (no_space and not prev.punct and "\u4e00" <= a[-1] <= "\u9fff"
            and b[0].isdigit()):
        return True
    return False


def _build_pieces(segments: list[dict]) -> list[_Piece]:
    """把所有 ASR 句拼成一条 token 流，并标记 ASR 句段末尾。"""
    pieces: list[_Piece] = []
    for seg in segments:
        tokens = build_tokens(seg, "zh")
        for k, tok in enumerate(tokens):
            pieces.append(_Piece(token=tok, segment_end=(k == len(tokens) - 1)))
    # 单句内部已经规整过；跨 ASR 句的闭/开引号要再收一次
    normalize_token_stream([p.token for p in pieces])
    return pieces


def _split_chunks(pieces: list[_Piece]) -> list[list[_Piece]]:
    """按硬断点切成可独立求解的块：句末标点或长静音之后必须断开。"""
    chunks: list[list[_Piece]] = []
    cur: list[_Piece] = []
    quote_depth = 0
    for i, p in enumerate(pieces):
        cur.append(p)
        tok = p.token
        quote_depth += sum(ch in _QUOTE_OPEN for ch in tok.text)
        quote_depth -= sum(ch in _QUOTE_CLOSE for ch in tok.text)
        quote_depth = max(0, quote_depth)
        is_sentence_end = _has_sentence_end(tok.punct)
        next_begin = pieces[i + 1].token.begin if i + 1 < len(pieces) else None
        long_gap = (next_begin is not None and next_begin - tok.end >= _HARD_GAP_MS)
        # 引语没闭合时，ASR 句号可能只是引语内部的停顿，不能把引号劈成两条
        if (is_sentence_end or long_gap) and quote_depth <= 0:
            chunks.append(cur)
            cur = []
            quote_depth = 0
    if cur:
        chunks.append(cur)
    return chunks


class _Scorer:
    """给区间 [i, j) 打分；分值越低，作为一条字幕越合适。"""

    def __init__(self, pieces: list[_Piece], preset: Preset):
        self.pieces = pieces
        self.p = preset
        self.n = len(pieces)
        self.tok = [x.token for x in pieces]
        self.pre = [0] * (self.n + 1)
        self.seg_pre = [0] * (self.n + 1)
        self.quote_after = [0] * (self.n + 1)
        depth = 0
        for i, piece in enumerate(pieces):
            self.pre[i + 1] = self.pre[i] + text_width(piece.token.text)
            self.seg_pre[i + 1] = self.seg_pre[i] + int(piece.segment_end)
            depth += sum(ch in _QUOTE_OPEN for ch in piece.token.text)
            depth -= sum(ch in _QUOTE_CLOSE for ch in piece.token.text)
            self.quote_after[i + 1] = max(0, depth)

        # 意群保护：短到本来就放得下的意群，内部切点设为不可达。
        # 这样“默认/沉浸”不会为了凑目标字数，把“这个方法非常适用，”切成两半。
        self.protected: set[int] = set()
        if preset.protect_clause:
            start = 0
            for j in range(1, self.n + 1):
                piece = self.pieces[j - 1]
                end_clause = (_has_clause_end(piece.token.punct)
                              or _has_sentence_end(piece.token.punct)
                              or piece.segment_end)
                if not end_clause:
                    continue
                width = self.pre[j] - self.pre[start]
                duration = self.pieces[j - 1].token.end - self.pieces[start].token.begin
                if width <= preset.max_chars * 2 and duration <= preset.max_duration_ms:
                    self.protected.update(range(start + 1, j))
                start = j

        # 边界代价只与切点有关，先算一次。这样“语义上不该切”的位置
        # 不依赖左右区间长度，不会被填满率的平方项轻易买通。
        self.cut_cost = [0.0] * (self.n + 1)
        for j in range(1, self.n + 1):
            self.cut_cost[j] = 1e9 if j in self.protected else self._boundary_cost(j)

    def chars(self, i: int, j: int) -> float:
        return (self.pre[j] - self.pre[i]) / 2.0

    def _boundary_cost(self, j: int) -> float:
        p = self.p
        prev = self.tok[j - 1]
        tail = prev.punct
        c = 0.0
        if _has_sentence_end(tail):
            c -= 220.0
        elif _has_clause_end(tail):
            c -= p.clause_reward
        elif self.pieces[j - 1].segment_end:
            c -= p.segment_reward
        else:
            c += p.cut_penalty

        prev_core = _core(prev.text)
        ends_with_stop = bool(tail) and (_has_clause_end(tail) or _has_sentence_end(tail))
        if not ends_with_stop:
            if _ends_with_bad_function(prev.text):
                c += 5000.0
            if prev_core in _BAD_END_WORDS:
                c += 5000.0
            if _ends_with_struct_particle(prev.text):
                c += 5000.0

        if j < self.n:
            nxt = self.tok[j]
            head = nxt.text.lstrip()
            if head and head[0] in _BAD_START:
                c += p.bad_start_penalty
            if head and _is_struct_particle(head):
                c += 5000.0
            if head and _core(head).startswith(_BAD_START_WORDS):
                c += 5000.0
            if head and _starts_with_left_particle(head):
                c += 10000.0
            if _bad_lexical_boundary(prev, nxt):
                c += 10000.0
            if _verb_object_boundary(prev, nxt):
                c += 800.0
            end_core = _core(prev.text)
            if end_core and end_core[-1] in _CLASSIFIER_END and _core(head):
                c += 10000.0
            if prev.text.rstrip()[-1:] in _OPEN_TAIL or head[:1] in _CLOSE_HEAD:
                c += 10000.0
            if self.quote_after[j] > 0:
                c += 10000.0       # 引语内部不主动断开；只有超限时才被迫切
        return c

    def __call__(self, i: int, j: int) -> float:
        if i >= j:
            return _INF
        p = self.p
        chars = self.chars(i, j)
        if chars > p.max_chars:
            return _INF
        dur = self.tok[j - 1].end - self.tok[i].begin
        if dur > p.max_duration_ms:
            return _INF

        # 主项：偏离目标长度越远越差。目标不是“必须填满”，而是排版锚点。
        c = (chars - p.target_chars) ** 2 * 4.0

        # 太短：一两个字独立成条会闪，也会打断语义
        if chars < p.min_chars:
            c += (p.min_chars - chars) ** 2 * 90.0

        # 太短/太快：观众读不完或来不及反应
        if dur < p.min_duration_ms:
            c += ((p.min_duration_ms - dur) / 1000.0) ** 2 * 900.0
        if dur > 0:
            cps = chars / (dur / 1000.0)
            if cps > p.max_cps:
                c += (cps - p.max_cps) ** 2 * 90.0

        c += self.cut_cost[j]

        # 跨 ASR 句段合并是允许的，但代价更高：优先在句段边界收束
        inner_segments = self.seg_pre[j - 1] - self.seg_pre[i]
        if inner_segments:
            c += inner_segments * p.cross_segment_penalty
        return c


def _dp_split(pieces: list[_Piece], preset: Preset) -> list[int]:
    """在一个硬断块内求最小代价切分，返回切点。"""
    n = len(pieces)
    if not n:
        return [0]
    scorer = _Scorer(pieces, preset)
    f = [_INF] * (n + 1)
    back = [-1] * (n + 1)
    f[0] = 0.0
    for j in range(1, n + 1):
        best, bi = _INF, -1
        for i in range(j - 1, -1, -1):
            if f[i] == _INF:
                continue
            c = scorer(i, j)
            if c == _INF:
                # 宽度/时长随 i 前移只增不减，继续往前不会更可行
                break
            if f[i] + c < best:
                best, bi = f[i] + c, i
        if bi < 0:
            # 单个 token 自身就超限：只能独占一条，避免无解
            bi = j - 1
            best = f[bi] + (max(0, scorer.chars(bi, j) - preset.max_chars) ** 2) * 20
        f[j], back[j] = best, bi
    cuts, j = [], n
    while j > 0:
        cuts.append(back[j])
        j = back[j]
    cuts.reverse()
    return cuts


def _chunk_cues(chunk: list[_Piece], preset: Preset, stats: dict | None) -> list[dict]:
    if len(chunk) > _MAX_TOKENS_PER_DP:
        # 极端长句：保守地按块处理，块边界作为硬切点
        out: list[dict] = []
        for s in range(0, len(chunk), _MAX_TOKENS_PER_DP):
            out.extend(_chunk_cues(chunk[s:s + _MAX_TOKENS_PER_DP], preset, stats))
        return out

    cuts = _dp_split(chunk, preset)
    out: list[dict] = []
    for a, b in zip(cuts, cuts[1:] + [len(chunk)]):
        if b <= a:
            continue
        toks = [x.token for x in chunk[a:b]]
        text = "".join(t.text for t in toks).strip()
        if not text:
            continue
        cue = {"text": text, "begin": toks[0].begin, "end": toks[-1].end}
        out.append(cue)
        if stats is not None:
            chars = text_width(text) / 2.0
            if chars > preset.max_chars:
                stats["over_limit"] = stats.get("over_limit", 0) + 1
    return out


def split_by_preset(segments: list[dict], preset_code: str = DEFAULT_PRESET,
                    stats: dict | None = None) -> list[dict]:
    """按预设切字幕。punct 直断；其余三档共用同一确定性引擎。"""
    if preset_code == "punct":
        return split_by_punct(segments, "zh")
    preset = PRESETS.get(preset_code)
    if preset is None:
        raise ValueError(f"未知断句档位: {preset_code}")

    pieces = _build_pieces(segments)
    out: list[dict] = []
    for chunk in _split_chunks(pieces):
        out.extend(_chunk_cues(chunk, preset, stats))
    return finalize(out, min_duration_ms=preset.min_duration_ms)
