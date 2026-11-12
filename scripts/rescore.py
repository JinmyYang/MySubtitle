"""离线重评分：用 results/eval 下已生成的 SRT 重新计算指标并更新报告（不调 API）。

用法: python scripts/rescore.py

素材、参考字幕与 SRT 存放位置全部从 scripts/evaluate.py 的 discover_cases() /
case_key() 推导，因此 evaluate.py 换素材、改归一化规则之后这里跟着走即可——
之前它硬引用 CASES 常量与 results/eval/<语种>/<模型>.srt，evaluate.py 一重构就失联了。
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, srt  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    EVAL_OUT, alignment_score, case_key, cer, discover_cases, normalize_zh,
    strip_numbers,
)


def main() -> int:
    models = list(config.MODELS)
    cutoff_ms = 15 * 60_000 if shutil.which("ffmpeg") else float("inf")

    cases = discover_cases()
    if not cases:
        print("未发现任何素材：测试音频/ 下需要「普通话音频 + 同名参考字幕（.zh.srt）」")
        return 1

    # (语种, 模型) -> 各素材的评分列表；同语种多素材时求平均，不再互相覆盖
    stats: dict[tuple[str, str], list[dict]] = {}
    for case in cases:
        ref_segs = [s for s in srt.parse_srt(case["ref"]) if s["begin"] < cutoff_ms]
        ref_text = " ".join(s["text"] for s in ref_segs)
        for model in models:
            path = EVAL_OUT / case_key(case) / f"{model}.srt"
            if not path.exists():
                continue
            hyp_segs = srt.parse_srt(path)
            hyp_text = " ".join(s["text"] for s in hyp_segs)
            align = alignment_score(ref_segs, hyp_segs)
            score = cer(normalize_zh(strip_numbers(ref_text)),
                        normalize_zh(strip_numbers(hyp_text)))
            stats.setdefault(("zh", model), []).append(
                {"score": score, "align": align})
            print(f"{case_key(case)} | {model:<28} "
                  f"CER = {score:.2%} | 时间戳平均偏差 {align:.2f}s")

    ordered = ["zh"]
    lines = [
        "# 模型评测报告",
        "",
        "- 音频窗口：前 15 分钟（参考字幕截取相同窗口；无 ffmpeg 时用完整音频）",
        "- 参考字幕：测试音频/ 下各素材自带的字幕，用于算 CER，不做断句质量基准",
        "- 普通话指标 CER（字符错误率），越低越好",
        "- 已做数字归一化（300 与 三百 等价），并移除括号内舞台说明",
        "- 时间戳平均偏差：参考句中心点到最近识别句中心点的平均距离（秒，越小越准）",
        "- 注意：ASR 会忠实记下口语填充词（啊/呢/这个/就是…），参考字幕是清理过的，",
        "  所以中文 CER 里插入错误占比偏高，不能简单等同于「认错了字」",
        "",
        "| 模型 | 普通话 CER | 平均 | 时间戳偏差 |",
        "|" + "---|" * (len(ordered) + 3),
    ]
    best: tuple[str, float] | None = None
    for model in models:
        cells, scores, aligns = [], [], []
        for lang in ordered:
            rows = stats.get((lang, model)) or []
            if rows:
                avg_lang = sum(r["score"] for r in rows) / len(rows)
                cells.append(f"{avg_lang:.2%}")
                scores.append(avg_lang)
                aligns.append(" / ".join(f"{r['align']:.1f}s" for r in rows))
            else:
                cells.append("-")
                aligns.append("-")
        avg = sum(scores) / len(scores) if scores else float("inf")
        if scores and (best is None or avg < best[1]):
            best = (model, avg)
        cells.append(f"{avg:.2%}" if scores else "-")
        lines.append(f"| {model} | {' | '.join(cells)} | {' / '.join(aligns)} |")
    lines.append("")
    if best:
        lines.append(f"**推荐默认模型：`{best[0]}`（平均错误率 {best[1]:.2%}）**")
    lines.append("")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}")
    rel = EVAL_OUT.relative_to(EVAL_OUT.parent.parent).as_posix()
    lines.append(f"各模型生成的 SRT 保存在 `{rel}/<素材名>_<语种>/<模型>.srt`，可人工对比细节。")

    report = EVAL_OUT / "evaluation_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n报告已更新: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
