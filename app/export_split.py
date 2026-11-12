"""断句门面：按档位分发到「标点直断」或统一排版引擎。

对外只有四个档位：
  punct      -> split_punct.py    每个小句一条，完全尊重 ASR 标点
  default    -> split_presets.py  默认节奏
  frequent   -> split_presets.py  短视频节奏
  immersive  -> split_presets.py  内容向节奏

不再暴露“每段最大字数”，也不再调用 LLM。三种节奏共用同一套确定性求解器，
档位之间只差阅读目标，避免多份实现各自漂移。
"""
from __future__ import annotations

from .split_presets import DEFAULT_PRESET, PRESETS, split_by_preset
from .split_punct import split_by_punct
from .tokenize import is_punct_char as _is_punct_char
from .tokenize import text_width as _text_width


def split_segments_by_preset(segments: list[dict], preset: str = DEFAULT_PRESET,
                             stats: dict | None = None) -> list[dict]:
    """按档位生成字幕；未知档位直接报错，不静默降级。"""
    return split_by_preset(segments, preset, stats)
