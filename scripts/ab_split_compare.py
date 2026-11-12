"""重构前的 A/B 对比脚本：旧「标点硬切 + 贪心装填 + 等分」vs「token 级 DP 全局最优」。

⚠️ 这是历史证据脚本，日常验证请用 scripts/regress_export.py。
⚠️ 两处会误导的地方，跑之前务必知道：
   1. 这里的"旧策略"是按旧实现逻辑**复刻**的，不是旧代码本身。
      严格对比应从 git 历史取出重构前的 app/export_split.py 再跑。
   2. 这里的 DP 是**早期简化版**（句内 DP，无跨短句聚组、无阅读速度项）。
      最终引擎还多了这两项，实测半空率远低于本脚本的结果。
      所以本脚本只能说明"DP 优于等分"，不能用它的数字代表当前引擎水平。

它同时验证一个前提：ASR 词序列与原文能否用 O(n) 双指针精确对齐。
结论是能（中文 94%），所以旧实现里那层 SequenceMatcher 对齐是多余的。

只读项目数据，不改动任何业务代码。用法: python scripts/ab_split_compare.py
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_DUR = 6000          # 单条字幕最长时长，与 export_split 一致
INF = float("inf")


# ---------------- 基础工具 ----------------

def _cw(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def width(s: str) -> int:
    return sum(_cw(c) for c in s)


# 词内标点：不是断句点，只作为词的一部分（与 export_split._is_clause_punct 口径一致）
_IN_WORD_PUNCT = set("'\"`~-–—")


def is_punct(ch: str) -> bool:
    if ch in _IN_WORD_PUNCT:
        return False
    return unicodedata.category(ch)[0] == "P"


def is_punct_token(s: str) -> bool:
    t = s.strip()
    return bool(t) and all(unicodedata.category(c)[0] == "P" for c in t)


# ---------------- O(n) 双指针对齐 ----------------

def align(text: str, words: list[dict]):
    """把原文与 ASR 词序列做 O(n) 双指针精确对齐。

    返回 (punct_after, ok)：
      punct_after[k] = 第 k 个词之后紧跟的标点串（标点归属于前一个词）
      ok = 是否整句精确对齐（无回溯、无模糊匹配）
    失败返回 (None, False)。
    """
    ti, n = 0, len(text)
    punct_after: list[str] = []
    for w in words:
        while ti < n and text[ti].isspace():
            ti += 1
        for c in w["text"]:
            if c.isspace():
                continue
            if ti >= n or text[ti] != c:
                return None, False
            ti += 1
        p = ""
        while ti < n and is_punct(text[ti]):
            p += text[ti]
            ti += 1
        punct_after.append(p)
    while ti < n and text[ti].isspace():
        ti += 1
    return punct_after, ti == n


# ---------------- 策略 A：现有实现（标点硬切 -> 贪心装填 -> 超长等分） ----------------

def _clauses(words: list[dict], punct_after: list[str]) -> list[list[int]]:
    """标点处硬切子句，返回词下标区间列表。标点少的语言（中文）会退化成整句一个子句。"""
    out, cur = [], []
    for i in range(len(words)):
        cur.append(i)
        if punct_after[i] or is_punct_token(words[i]["text"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _rule_even(idxs: list[int], words: list[dict], limit: int) -> list[int]:
    """等分：N = ceil(宽/限宽)，目标落在等宽位置，吸附到最近词边界。"""
    total = sum(width(words[i]["text"]) for i in idxs)
    n = max(2, -(-total // limit))
    if n < 2:
        return []
    targets, acc, ti = [], 0, 1
    for i in idxs:
        acc += width(words[i]["text"])
        while ti < n and acc >= total * ti / n:
            targets.append(i + 1)
            ti += 1
    cuts, prev = [], idxs[0]
    for tp in targets:
        cands = [c for c in idxs if prev < c < idxs[-1] + 1]
        if not cands:
            break
        best = min(cands, key=lambda c: abs(c - tp))
        if best > prev:
            cuts.append(best)
            prev = best
    return cuts


def strategy_current(segs: list[dict], limit: int) -> list[tuple[int, list[int]]]:
    cues: list[tuple[int, list[int]]] = []
    for si, s in enumerate(segs):
        words, pa = s["words"], s["punct_after"]
        for cl in _clauses(words, pa):
            w = sum(width(words[i]["text"]) for i in cl)
            dur = words[cl[-1]]["end"] - words[cl[0]]["begin"]
            if w <= limit and dur <= MAX_DUR:
                pieces = [cl]
            else:
                cuts = _rule_even(cl, words, limit)
                bounds = [cl[0]] + cuts + [cl[-1] + 1]
                pieces = [list(range(bounds[k], bounds[k + 1]))
                          for k in range(len(bounds) - 1)]
                pieces = [p for p in pieces if p]
            buf: list[list[int]] = []
            for p in pieces:
                if buf:
                    bw = sum(width(words[i]["text"]) for q in buf for i in q) \
                        + sum(width(words[i]["text"]) for i in p)
                    bd = words[p[-1]]["end"] - words[buf[0][0]]["begin"]
                    if bw > limit or bd > MAX_DUR:
                        cues.extend((si, q) for q in buf)
                        buf = []
                buf.append(p)
            cues.extend((si, q) for q in buf)
    return cues


# ---------------- 策略 B：token 级 DP 全局最优 ----------------

def strategy_dp(segs: list[dict], limit: int, punct_bonus: int = 900,
                lead_penalty: int = 400) -> list[tuple[int, list[int]]]:
    """在一个 ASR 句内做 DP：切点只能是词下标，硬约束（限宽/最长时长）内建于代价函数，
    标点与虚词只作为代价特征参与权衡，不做硬切。"""
    cues: list[tuple[int, list[int]]] = []
    for si, s in enumerate(segs):
        words, pa = s["words"], s["punct_after"]
        n = len(words)
        pre = [0] * (n + 1)
        for i, w in enumerate(words):
            pre[i + 1] = pre[i] + width(w["text"])

        def cost(i: int, j: int) -> float:
            if i >= j:
                return INF
            w = pre[j] - pre[i]
            if w > limit:
                return INF
            if words[j - 1]["end"] - words[i]["begin"] > MAX_DUR:
                return INF
            c = float((limit - w) ** 2)      # 越接近限宽越省行
            if pa[j - 1]:                     # 以标点收尾 -> 天然停顿
                c -= punct_bonus
            if j < n and pa[j - 1] and is_punct_token(words[j]["text"]):
                c += lead_penalty             # 下一行以标点开头 -> 突兀
            return c

        f = [INF] * (n + 1)
        back = [-1] * (n + 1)
        f[0] = 0.0
        for j in range(1, n + 1):
            best, bi = INF, -1
            for i in range(j - 1, -1, -1):
                if f[i] == INF:
                    continue
                c = cost(i, j)
                if c == INF:
                    break                     # 宽度单调递增，再往前更宽
                if f[i] + c < best:
                    best, bi = f[i] + c, i
            f[j], back[j] = best, bi
        if f[n] == INF:
            cues.append((si, list(range(n))))
            continue
        cuts, j = [], n
        while j > 0:
            cuts.append(back[j])
            j = back[j]
        cuts.reverse()
        for a, b in zip(cuts, cuts[1:] + [n]):
            cues.append((si, list(range(a, b))))
    return cues


# ---------------- 评估 ----------------

def evaluate(name: str, cue_idxs: list[tuple[int, list[int]]],
             segs: list[dict], limit: int) -> dict:
    widths = [sum(width(w["text"]) for w in segs[si]["words"][idxs[0]:idxs[-1] + 1])
              for si, idxs in cue_idxs]
    n = len(widths)
    mean = sum(widths) / n
    sd = (sum((x - mean) ** 2 for x in widths) / n) ** 0.5
    tiny = sum(1 for x in widths if x < limit * 0.5)
    print(f"  {name:<12} 条数={n:>4}  平均宽={mean:5.1f}  标准差={sd:5.2f}  "
          f"限宽利用率={mean/limit:5.1%}  半空条={tiny:>3} ({tiny/n:5.1%})")
    return {"n": n, "mean": mean, "sd": sd, "util": mean / limit, "tiny": tiny / n}


def load(path: Path) -> tuple[list[dict], int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    segs, ok_n, total = [], 0, 0
    for s in raw:
        ws = [w for w in (s.get("words") or []) if (w.get("text") or "").strip()]
        if not ws:
            continue
        pa, ok = align(s["text"], ws)
        total += 1
        if ok:
            ok_n += 1
            segs.append({"words": ws, "punct_after": pa})
    return segs, ok_n, total


def run(path: Path, label: str, limits: list[int]) -> None:
    segs, ok_n, total = load(path)
    nwords = sum(len(s["words"]) for s in segs)
    print(f"\n{'='*94}")
    print(f"{label}   {total} 句 / {nwords} 词   "
          f"双指针精确对齐成功 {ok_n}/{total} ({ok_n/total:.0%})")
    for limit in limits:
        print(f"  --- 限宽 {limit} ---")
        evaluate("现有(等分)", strategy_current(segs, limit), segs, limit)
        evaluate("DP(最优)", strategy_dp(segs, limit), segs, limit)


# 按目录发现识别结果：样本不进版本库，写死文件名迟早对不上
SOURCES = (ROOT / "samples", ROOT / "assets" / "segments")
# 档位是"显示宽度"：中文 24/40 即 12/20 字
LIMITS = [24, 40]


def discover() -> list[Path]:
    out: list[Path] = []
    for folder in SOURCES:
        if folder.is_dir():
            out.extend(sorted(folder.glob("*.json")))
    return out


if __name__ == "__main__":
    paths = discover()
    if not paths:
        print("未找到识别结果数据（samples/ 或 assets/segments/）。")
        raise SystemExit(1)
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path.name}: {exc}")
            continue
        if not isinstance(raw, list) or not raw:
            print(f"[skip] {path.name}: 不是识别结果数组")
            continue
        text = "".join(str(s.get("text") or "") for s in raw)
        if sum("\u4e00" <= ch <= "\u9fff" for ch in text) < 20:
            print(f"[skip] {path.name}: 非普通话素材")
            continue
        run(path, f"{path.stem[:32]} (zh)", LIMITS)
