"""项目配置：API Key、模型注册表、路径常量。"""
from __future__ import annotations

import os
from pathlib import Path


def _load_api_key() -> str:
    """读取百炼 API Key。

    除环境变量外，Windows 上再回退读一次用户环境变量：用户在"系统属性"里配好 Key 后，
    已打开的终端和新起的进程经常读不到（本项目的实测情况），结果任务直接报未配置。
    这里兜一层，服务重启后即可生效。
    """
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key:
        return key
    if os.name != "nt":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, "DASHSCOPE_API_KEY")
            return (val or "").strip()
    except Exception:  # noqa: BLE001 - 读不到就当作未配置
        return ""


# 阿里云百炼 API Key（北京地域）
API_KEY = _load_api_key()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"   # 上传暂存区，入库后文件会被移走
RESULT_DIR = BASE_DIR / "results"   # 导出暂存区，入库后文件会被移走
ASSET_DIR = BASE_DIR / "assets"     # 资产库：音频 / 识别中间产物 / 字幕

# 百炼 REST 端点（北京地域）
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com"
UPLOADS_URL = f"{DASHSCOPE_BASE}/api/v1/uploads"
TRANSCRIPTION_URL = f"{DASHSCOPE_BASE}/api/v1/services/audio/asr/transcription"
TASK_URL = f"{DASHSCOPE_BASE}/api/v1/tasks"

# 上传文件大小上限（百炼临时存储限制 1GB）
MAX_UPLOAD_BYTES = 1 * 1024 * 1024 * 1024

ALLOWED_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".amr", ".wma",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".mpeg",
}

# 断句档位：不暴露字数参数，只给场景化目标。
SPLIT_PRESETS = [
    {"code": "punct", "name": "按标点", "desc": "每个小句一条字幕"},
    {"code": "default", "name": "默认", "desc": "大多数视频，完整与节奏平衡"},
    {"code": "frequent", "name": "频繁", "desc": "短视频，更短、更快、更跟手"},
    {"code": "immersive", "name": "沉浸", "desc": "内容向，少跳字，保留完整意群"},
]
DEFAULT_PRESET = "default"

# 模型注册表：只保留普通话实测效果最稳的 Fun-ASR。
MODELS: dict[str, dict] = {
    "fun-asr": {
        "name": "Fun-ASR",
        "default": True,
    },
}


def default_model() -> str:
    return next(iter(MODELS))


def build_language_params(model: str) -> dict:
    """固定按普通话生成模型语言参数。"""
    return {"language_hints": ["zh"]}
