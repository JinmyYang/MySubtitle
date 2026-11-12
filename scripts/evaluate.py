"""模型对比评测：把各普通话离线转写模型的输出与视频自带字幕对比 CER。

⚠️ 非 dry-run 会调用真实云端 API 并产生费用（fun-asr 约 0.79 元/小时音频）。
   先跑 --dry-run 看会处理哪些素材。本脚本只覆盖云端引擎。

素材不再硬编码：扫描 RESOURCES 目录，按文件名自动配对 (音频, 参考字幕)。
只评测带 `.zh.srt` 参考字幕的普通话素材，这样新增素材时不用改代码；
缺音频或缺字幕的会被自动跳过。

用法：
    python scripts/evaluate.py --dry-run           # 只看要跑哪些素材（不调 API）
    python scripts/evaluate.py                     # 全部模型，音频裁剪到前 15 分钟（需 ffmpeg）
    python scripts/evaluate.py --models fun-asr
    python scripts/evaluate.py --max-minutes 0     # 不裁剪，跑完整音频

注意：非 dry-run 会真实调用百炼转写接口并产生费用。
"""
from __future__ import annotations

import argparse
import glob
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, dashscope_client, srt  # noqa: E402

RESOURCES = Path(__file__).resolve().parent.parent / "测试音频"
EVAL_OUT = config.RESULT_DIR / "eval"

# ⚠️ 参考字幕的用途边界（很重要）：
#   这些字幕来自视频网站（人工或自动生成），**文本是准确的**，所以可以当 ground truth
#   算 CER，用来衡量「ASR 认错了多少字」。
#   但它们的断句是人工/网站按自己的风格打的，**不是好的字幕断句范本**，
#   绝不能用参考字幕的时间轴或断行位置来评价本项目断句引擎的好坏。
#   断句质量请用 scripts/regress_export.py 的限宽利用率 / 半空条占比，或肉眼判断。


def _find(pattern: str) -> Path:
    hits = sorted(RESOURCES.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"在 {RESOURCES} 找不到匹配 {pattern} 的文件")
    return hits[0]


# 参考字幕名里的普通话标记：xxx.zh.srt / xxx.zh-Hans.srt
_ZH_IN_NAME = re.compile(r"\.zh(?:[-_][A-Za-z]+)*\.srt$", re.I)


def discover_cases() -> list[dict]:
    """扫描素材目录，自动配对 (普通话音频, 参考字幕)。"""
    if not RESOURCES.exists():
        return []
    cases: list[dict] = []
    for audio in sorted(RESOURCES.glob("*.m4a")):
        refs = [srt for srt in sorted(RESOURCES.glob(f"{glob.escape(audio.stem)}*.srt"))
                if _ZH_IN_NAME.search(srt.name)]
        if refs:
            cases.append({"lang": "zh", "audio": audio, "ref": refs[0]})
    return cases


def case_key(case: dict) -> str:
    """评测单元的唯一标识（同语种可能有多个素材，不能只用 lang 当 key）。"""
    return f"{case['audio'].stem[:24]}_{case['lang']}"


# ---------------- 文本归一化与指标 ----------------

_DECOR_RE = re.compile(r"[\[\(（【｛<{][^\]\)）】｝>}]*[\]\)）】｝>}]", re.S)


def normalize_zh(text: str) -> str:
    text = _DECOR_RE.sub("", text)
    return "".join(ch.lower() for ch in text if unicodedata.category(ch)[0] not in "PSZ")


def levenshtein(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b)


def cer(ref: str, hyp: str) -> float:
    """中文：字符错误率。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


# ---------------- 数字归一化（避免“300”/“三百”写法差异干扰指标） ----------------

_DIGIT_UNITS = ["亿", "万", "千", "百", "十"]
_DIGIT_CHARS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_num_to_int(s: str) -> int | None:
    """简单中文数字转整数，失败返回 None。"""
    if s in _DIGIT_CHARS:
        return _DIGIT_CHARS[s]
    if "亿" in s or "万" in s:
        return None  # 大数少見，直接丢弃该数字串即可（两侧同规则）
    result, num = 0, 0
    for ch in s:
        if ch in _DIGIT_CHARS:
            num = _DIGIT_CHARS[ch]
        elif ch in _DIGIT_UNITS:
            unit = {"千": 1000, "百": 100, "十": 10}[ch]
            result += (num if num else 1) * unit
            num = 0
        else:
            return None
    return result + num


_NUM_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*[%]?" +
                           "|" + f"[零一二两三四五六七八九十百千万亿]+")


def strip_numbers(text: str) -> str:
    """移除/归一化数字，使 300 与 三百 等价（统一转为阿拉伯数字后删除）。"""
    def repl(m: re.Match) -> str:
        s = m.group(0)
        if s[0].isdigit():
            return ""
        val = _cn_num_to_int(s)
        return "" if val is not None else s
    return _NUM_TOKEN_RE.sub(repl, text)


# ---------------- 时间戳对齐质量 ----------------

def alignment_score(ref_segs: list[dict], hyp_segs: list[dict]) -> float:
    """参考字幕句的时间中心点在假设字幕时间线上的平均绝对偏差（秒）。
    对重叠的自动字幕（参考句互相重叠）仅作粗略参考。"""
    if not hyp_segs or not ref_segs:
        return float("inf")
    hyp_ranges = [((s["begin"] + s["end"]) / 2000.0,
                   s["begin"] / 1000.0, s["end"] / 1000.0) for s in hyp_segs]
    total, n = 0.0, 0
    for s in ref_segs:
        c = (s["begin"] + s["end"]) / 2000.0
        best = min((abs(c - hc) for hc, _, _ in hyp_ranges), default=0.0)
        total += best
        n += 1
    return total / n if n else float("inf")


# ---------------- 音频裁剪 ----------------

def trim_audio(src: Path, minutes: int) -> Path:
    """用 ffmpeg 截取前 N 分钟（-c copy 快速）。无 ffmpeg 或 minutes<=0 时返回原文件。"""
    if minutes <= 0 or shutil.which("ffmpeg") is None:
        return src
    out = config.RESULT_DIR / "eval" / "clips" / f"{minutes}min_{src.stem}{src.suffix}"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-t", str(minutes * 60), "-c", "copy", str(out)],
        check=True,
    )
    return out


# ---------------- 主流程 ----------------

def run_case_model(case: dict, model: str, audio: Path) -> dict:
    """单条音频×单个模型：上传->转写->生成 SRT。返回任务信息。"""
    t0 = time.time()
    url = dashscope_client.upload_file(model, audio)
    task_id = dashscope_client.submit_transcription(model, url)
    output = dashscope_client.wait_for_task(task_id, poll_interval=3.0)
    segments = dashscope_client.fetch_segments(output)

    # 同一素材可能有多语种参考字幕，用 stem 区分，避免互相覆盖
    out_path = EVAL_OUT / case_key(case) / f"{model}.srt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(srt.segments_to_srt(segments), encoding="utf-8")
    return {"elapsed": time.time() - t0, "segments": segments, "srt_path": out_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(config.MODELS))
    parser.add_argument("--max-minutes", type=int, default=15,
                        help="每条音频只评测前 N 分钟，0 表示完整音频")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出将要评测的素材与模型，不调用 API（评测是付费的）")
    parser.add_argument("--only", default="",
                        help="只评测文件名包含该子串的素材")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in config.MODELS:
            sys.exit(f"未知模型: {m}")

    cases = discover_cases()
    if args.only:
        cases = [c for c in cases if args.only in c["audio"].name]
    if not cases:
        sys.exit(f"在 {RESOURCES} 下没有找到 (音频, 参考字幕) 配对；"
                 f"需要同名的 .m4a 与带语种标记的 .srt（如 xxx.zh.srt）")

    print(f"素材目录: {RESOURCES}")
    print(f"待评测配对 {len(cases)} 组 / 模型 {len(models)} 个：")
    for c in cases:
        print(f"  [{c['lang']}] {c['audio'].name}")
        print(f"        参考: {c['ref'].name}")
    if args.dry_run:
        print("\n--dry-run：未调用任何接口。")
        return

    trimmed = args.max_minutes > 0 and shutil.which("ffmpeg") is not None
    # 参考字幕也需截取到与音频相同的时长窗口，否则指标失真
    cutoff_ms = args.max_minutes * 60_000 if trimmed else float("inf")

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, str], dict] = {}

    for case in cases:
        audio = trim_audio(case["audio"], args.max_minutes)
        print(f"\n=== {case['lang']} | {audio.name} ===")
        ref_segs = [s for s in srt.parse_srt(case["ref"]) if s["begin"] < cutoff_ms]
        ref_text = " ".join(s["text"] for s in ref_segs)

        for model in models:
            print(f"  -> {model} ...", flush=True)
            try:
                info = run_case_model(case, model, audio)
            except Exception as exc:  # noqa: BLE001
                print(f"     [fail] {exc}")
                results[(case_key(case), model)] = {"error": str(exc)}
                continue

            hyp_text = " ".join(s["text"] for s in info["segments"])
            align = alignment_score(ref_segs, info["segments"])
            score = cer(normalize_zh(strip_numbers(ref_text)),
                        normalize_zh(strip_numbers(hyp_text)))
            metric = "CER"
            results[(case_key(case), model)] = {
                "score": score, "metric": metric, "align": align,
                "elapsed": info["elapsed"], "n_segments": len(info["segments"]),
            }
            print(f"     {metric} = {score:.2%} | 时间戳平均偏差 {align:.2f}s "
                  f"| 耗时 {info['elapsed']:.0f}s "
                  f"| {len(info['segments'])} 句 -> {info['srt_path'].name}")

    # ---------------- 报告 ----------------
    keys = [case_key(c) for c in cases]
    cols = [f"{c['audio'].stem[:18]} CER" for c in cases]
    lines = [
        "# 模型评测报告",
        "",
        f"- 音频窗口：{'前 %d 分钟' % args.max_minutes if trimmed else '完整音频'}",
        f"- 参考字幕：{RESOURCES} 下各视频自带字幕（截取相同窗口）",
        "- 普通话指标 CER（字符错误率），越低越好",
        "",
        "| 模型 | " + " | ".join(cols) + " | 平均 | 总耗时 |",
        "|" + "---|" * (len(cols) + 2),
    ]
    best = None
    for model in models:
        cells, scores = [], []
        total_t = 0.0
        for key in keys:
            r = results.get((key, model), {})
            if "error" in r:
                cells.append("失败")
            elif "score" in r:
                cells.append(f"{r['score']:.2%}")
                scores.append(r["score"])
                total_t += r["elapsed"]
            else:
                cells.append("-")
        avg = sum(scores) / len(scores) if scores else float("inf")
        if scores and (best is None or avg < best[1]):
            best = (model, avg)
        if scores:
            lines.append(f"| {model} | {' | '.join(cells)} | {avg:.2%} | {total_t:.0f}s |")
        else:
            lines.append(f"| {model} | {' | '.join(cells)} | - | - |")
    lines.append("")
    if best:
        lines.append(f"**推荐默认模型：`{best[0]}`（平均错误率 {best[1]:.2%}）**")
    lines.append("")
    lines.append("各模型生成的 SRT 保存在 `results/eval/<语种>/<模型>.srt`，可人工对比细节。")

    report = EVAL_OUT / "evaluation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n报告已保存: {report}")


if __name__ == "__main__":
    main()
