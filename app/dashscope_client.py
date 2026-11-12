"""阿里云百炼语音识别 REST 客户端。

流程：获取上传凭证 -> 上传文件到临时 OSS -> 提交异步转写任务 -> 轮询任务 -> 下载转写结果。
统一走 REST（httpx），避免 SDK 对 oss:// 临时 URL 支持不一致的问题。
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
import re

from . import config


class DashScopeError(RuntimeError):
    """百炼接口调用失败。"""


def _auth_headers(extra: dict | None = None) -> dict:
    if not config.API_KEY:
        raise DashScopeError("未找到 DASHSCOPE_API_KEY 环境变量，请先配置百炼 API Key")
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
    }
    if extra:
        headers.update(extra)
    return headers


def get_upload_policy(model: str) -> dict:
    """获取指定模型的临时存储上传凭证（文件与模型绑定）。"""
    resp = httpx.get(
        config.UPLOADS_URL,
        headers=_auth_headers(),
        params={"action": "getPolicy", "model": model},
        timeout=30,
    )
    if resp.status_code != 200:
        raise DashScopeError(f"获取上传凭证失败[{resp.status_code}]: {resp.text[:300]}")
    data = resp.json()
    if "data" not in data:
        raise DashScopeError(f"获取上传凭证返回异常: {resp.text[:300]}")
    return data["data"]


def upload_file(model: str, file_path: str | Path, progress_cb=None) -> str:
    """把本地文件上传到百炼临时存储，返回 oss:// 临时 URL。"""
    policy = get_upload_policy(model)
    file_path = Path(file_path)
    key = f"{policy['upload_dir']}/{file_path.name}"
    fields = {
        "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
        "Signature": (None, policy["signature"]),
        "policy": (None, policy["policy"]),
        "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
        "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
        "key": (None, key),
        "success_action_status": (None, "200"),
    }
    if progress_cb:
        progress_cb("oss_uploading")
    with file_path.open("rb") as fh:
        files = dict(fields)
        files["file"] = (file_path.name, fh)
        resp = httpx.post(policy["upload_host"], files=files, timeout=600)
    if resp.status_code != 200:
        raise DashScopeError(f"上传文件到临时存储失败[{resp.status_code}]: {resp.text[:300]}")
    return f"oss://{key}"


def build_transcription_payload(model: str, file_url: str) -> dict:
    """构造 Fun-ASR 非实时语音识别请求体。

    Fun-ASR 使用 file_urls 数组；词级时间戳固定开启，不需要 enable_words。
    """
    return {
        "model": model,
        "input": {"file_urls": [file_url]},
        "parameters": config.build_language_params(model),
    }


def submit_transcription(model: str, file_url: str) -> str:
    """提交普通话异步录音文件识别任务，返回 task_id。"""
    payload = build_transcription_payload(model, file_url)
    headers = _auth_headers({
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
        # 使用 oss:// 临时 URL 时必须携带
        "X-DashScope-OssResourceResolve": "enable",
    })
    resp = httpx.post(config.TRANSCRIPTION_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise DashScopeError(f"提交转写任务失败[{resp.status_code}]: {resp.text[:400]}")
    output = resp.json().get("output", {})
    task_id = output.get("task_id")
    if not task_id:
        raise DashScopeError(f"提交转写任务未返回 task_id: {resp.text[:400]}")
    return task_id


def query_task(task_id: str) -> dict:
    """查询任务状态，返回 output 字段。"""
    resp = httpx.get(
        f"{config.TASK_URL}/{task_id}",
        headers=_auth_headers(),
        timeout=30,
    )
    if resp.status_code != 200:
        raise DashScopeError(f"查询任务失败[{resp.status_code}]: {resp.text[:300]}")
    return resp.json().get("output", {})


def wait_for_task(task_id: str, poll_interval: float = 3.0, timeout: float = 3600,
                  progress_cb=None) -> dict:
    """轮询任务直到终态，返回最终 output。"""
    start = time.time()
    while True:
        output = query_task(task_id)
        status = output.get("task_status")
        if progress_cb:
            progress_cb(status)
        if status == "SUCCEEDED":
            return output
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            raise DashScopeError(
                f"转写任务{status}: {output.get('message') or output.get('code') or output}"
            )
        if time.time() - start > timeout:
            raise DashScopeError(f"转写任务超时（>{timeout:.0f}s）: task_id={task_id}")
        time.sleep(poll_interval)


# SenseVoice 等模型会在文本中夹带 <|Speech|>、<|Laughter|> 之类的事件标签，需要清理
_TAG_RE = re.compile(r"<\|[^|]*\|>")


def parse_transcription_data(data: dict) -> list[dict]:
    """把转写结果 JSON（transcripts 结构）解析为句级片段列表。"""
    segments: list[dict] = []
    for transcript in data.get("transcripts") or []:
        for sent in transcript.get("sentences") or []:
            text = _TAG_RE.sub("", sent.get("text") or "")
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            seg = {
                "begin": int(sent.get("begin_time") or 0),
                "end": int(sent.get("end_time") or 0),
                "text": text,
            }
            # 词级时间戳（Fun-ASR 固定返回），供断句器精确定时；
            # 标点符号词保留，作为断句点标记
            words = []
            for wd in sent.get("words") or []:
                wtxt = _TAG_RE.sub("", wd.get("text") or "")
                if not wtxt.strip():
                    continue
                words.append({
                    "text": wtxt,
                    "begin": int(wd.get("begin_time") or 0),
                    "end": int(wd.get("end_time") or 0),
                })
            if words:
                seg["words"] = words
            segments.append(seg)
    segments.sort(key=lambda s: s["begin"])
    return segments


def fetch_segments(output: dict) -> list[dict]:
    """从 Fun-ASR 任务结果中提取句级片段列表（毫秒）。"""
    url = ""
    results = output.get("results") or []
    if results:
        item = results[0]
        if item.get("subtask_status") not in ("SUCCEEDED", None):
            raise DashScopeError(f"音频转写子任务失败: {item}")
        url = item.get("transcription_url") or ""
    if not url:
        raise DashScopeError(f"转写结果缺少 transcription_url: {output}")
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()
    return parse_transcription_data(resp.json())
