"""对运行中的服务做真实 HTTP 端到端验证（调用真实百炼 API，音频约 2 分钟）。

⚠️ 会产生云端 API 费用（fun-asr 约 0.79 元/小时音频）。
   本地引擎不需要联网，直接在 Web 界面点一次识别就能验证。

刻意不带 test_ 前缀：它是一个手工执行的联网检查脚本，不是 pytest 用例
（放在 tests/ 下用 test_ 命名会让 pytest 收集到"零个用例"，造成有测试的错觉）。

前置：先跑 `python scripts/smoke_test.py` 生成 results/clips/smoke_zh_2min.m4a。

用法: python tests/live_http_check.py [http://127.0.0.1:18080]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
CLIP = Path(__file__).resolve().parent.parent / "results" / "clips" / "smoke_zh_2min.m4a"
MODEL = "fun-asr"


def _wait_job(client: httpx.Client, job_id: str) -> dict:
    info: dict = {}
    for _ in range(120):
        info = client.get(f"{BASE}/api/jobs/{job_id}", timeout=30).json()
        print(f"  状态: {info['status']} | 阶段: {info['stage_text']} | 进度: {info['progress']}%")
        if info["status"] in ("done", "failed"):
            return info
        time.sleep(3)
    return info


def _wait_export(client: httpx.Client, job_id: str, export_id: str) -> dict:
    export: dict = {}
    for _ in range(60):
        exports = client.get(f"{BASE}/api/jobs/{job_id}/exports", timeout=30).json()["exports"]
        export = next(e for e in exports if e["id"] == export_id)
        if export["status"] != "pending":
            return export
        time.sleep(1)
    return export


def main() -> int:
    if not CLIP.exists():
        print(f"缺少测试音频 {CLIP}\n请先运行: python scripts/smoke_test.py")
        return 1

    with httpx.Client() as client:
        with CLIP.open("rb") as fh:
            r = client.post(f"{BASE}/api/jobs",
                            files={"file": (CLIP.name, fh)},
                            data={"model": MODEL},
                            timeout=300)
        r.raise_for_status()
        job = r.json()
        print(f"已提交识别任务: {job['id']} ({job['model_name']})")

        info = _wait_job(client, job["id"])
        assert info["status"] == "done", f"识别失败: {info.get('error')}"
        print(f"识别完成，音频 {info['duration_sec']}s，预览首句: {info['preview'][0]['text']}")

        r = client.post(f"{BASE}/api/jobs/{job['id']}/exports",
                        data={"preset": "default"}, timeout=60)
        r.raise_for_status()
        export = r.json()
        print(f"已创建字幕导出: {export['id']} (默认档)")

        export = _wait_export(client, job["id"], export["id"])
        assert export["status"] == "done", f"导出失败: {export.get('error')}"
        print(f"导出完成，共 {export['cue_count']} 条")

        # 导出文件已收编进资产库，两条路径都要能下
        r = client.get(f"{BASE}/api/jobs/{job['id']}/exports/{export['id']}/download"
                       f"?format=srt", timeout=30)
        assert r.status_code == 200 and "-->" in r.text, f"导出下载失败: {r.status_code}"
        print(f"导出 SRT 下载成功（{len(r.text)} 字符）")

        r = client.get(f"{BASE}/api/assets/{export['asset_id']}/file", timeout=30)
        assert r.status_code == 200 and "-->" in r.text, f"资产下载失败: {r.status_code}"
        print(f"资产 SRT 下载成功（{len(r.text)} 字符）")

    print("\n真实链路验证通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
