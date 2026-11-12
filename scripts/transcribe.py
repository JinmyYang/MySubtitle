"""把本地音频/视频转写成分段（句级 + 词级时间戳），落盘为 segments.json。

⚠️ 会调用真实云端 API 并产生费用（fun-asr 约 0.79 元/小时音频）。
   本脚本只覆盖云端引擎；本地识别免费，用 Web 界面识别后从资产库导出即可。

用途：为断句引擎准备测试素材。落盘后可反复用于导出测试，无需重复付费转写。

用法:
    python scripts/transcribe.py 测试音频/普通话测试.m4a
    python scripts/transcribe.py 测试音频/x.m4a --model fun-asr
    python scripts/transcribe.py 测试音频/x.m4a --out samples/我的样本.json

说明：API Key 由 app.config 统一读取（环境变量取不到时会回退读 Windows 用户环境变量），
本脚本不重复实现。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, dashscope_client  # noqa: E402


def ensure_api_key() -> None:
    """config 在 import 时已读取 Key（含 Windows 用户变量回退）；这里只负责提示。"""
    if not config.API_KEY:
        sys.exit("未找到 DASHSCOPE_API_KEY。请在系统环境变量中配置百炼 API Key 后重试。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="音频/视频文件路径")
    ap.add_argument("--model", default=config.default_model())
    ap.add_argument("--out", default="", help="输出 json 路径；默认 samples/<文件名>_<模型>_segments.json")
    args = ap.parse_args()

    ensure_api_key()
    src = Path(args.audio)
    if not src.exists():
        sys.exit(f"找不到文件: {src}")
    if src.suffix.lower() not in config.ALLOWED_EXTS:
        sys.exit(f"不支持的格式: {src.suffix}")

    dst = (Path(args.out) if args.out
           else Path("samples") / f"{src.stem}_{args.model}_segments.json")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"音频: {src} ({src.stat().st_size / 1048576:.1f} MB) | 模型: {args.model}")
    print("1/3 上传到百炼临时存储...", flush=True)
    url = dashscope_client.upload_file(args.model, src)
    print("2/3 提交转写任务...", flush=True)
    task_id = dashscope_client.submit_transcription(args.model, url)
    print("       task_id:", task_id)
    print("3/3 等待转写完成...", flush=True)
    output = dashscope_client.wait_for_task(
        task_id, progress_cb=lambda s: print("       状态:", s, flush=True))
    segments = dashscope_client.fetch_segments(output)
    if not segments:
        sys.exit("转写结果为空")

    dst.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
    dur = segments[-1]["end"] / 1000
    print(f"\n完成：{len(segments)} 句，音频时长 {dur / 60:.1f} 分钟 -> {dst}")


if __name__ == "__main__":
    main()
