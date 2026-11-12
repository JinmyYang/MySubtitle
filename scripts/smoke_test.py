"""冒烟测试：用一小段普通话音频验证 上传 -> 转写 -> SRT 全链路（无头，不启动 Web 服务）。

用法:
    python scripts/smoke_test.py [model] [音频路径]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, dashscope_client, srt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESOURCES = ROOT / "测试音频"


def _pick_audio() -> Path:
    """自动挑一段普通话素材：优先中文文件名，避免误用历史英文素材。"""
    files = sorted(RESOURCES.glob("*.m4a"))
    chinese = [p for p in files if any("\u4e00" <= ch <= "\u9fff" for ch in p.stem)]
    if not chinese:
        raise SystemExit(f"在 {RESOURCES} 下没有找到普通话音频素材")
    return chinese[0]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "fun-asr"
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else _pick_audio()

    clip_dir = config.RESULT_DIR / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip = clip_dir / "smoke_zh_2min.m4a"
    if not clip.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-t", "120", "-c", "copy", str(clip)],
            check=True,
        )
    print(f"测试音频: {clip} ({clip.stat().st_size / 1048576:.1f} MB)")

    t0 = time.time()
    print("1/4 上传到百炼临时存储...")
    url = dashscope_client.upload_file(model, clip)
    print(f"      oss url: {url[:80]}...")

    print("2/4 提交转写任务...")
    task_id = dashscope_client.submit_transcription(model, url)
    print(f"      task_id: {task_id}")

    print("3/4 等待转写完成...")
    output = dashscope_client.wait_for_task(task_id, poll_interval=2.0)

    print("4/4 解析结果并生成 SRT...")
    segments = dashscope_client.fetch_segments(output)
    out = config.RESULT_DIR / "smoke_test.srt"
    out.write_text(srt.segments_to_srt(segments), encoding="utf-8")
    print(f"\n完成！耗时 {time.time() - t0:.0f}s，共 {len(segments)} 句，输出: {out}")
    for seg in segments[:5]:
        print(f"  [{seg['begin'] // 1000:>3}s] {seg['text']}")


if __name__ == "__main__":
    main()
