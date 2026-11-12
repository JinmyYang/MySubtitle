"""本地 Fun-ASR-Nano 识别后端。

模型按需加载、常驻复用，支持手动卸载释放显存（网页开着但暂时不识别时可以把
4GB 显存还给别的程序）。加载锁保证只加载一次，推理锁保证串行——并发进 GPU
会直接把 8GB 显存挤爆。

输出的 segments 与云端 Fun-ASR 同构（begin/end/text/words，单位毫秒），
可直接喂给现有断句引擎。

⚠️ 映射要点（实测结论，别改回去）：CTC 的 timestamps 里**自带标点 token**，
而 tokenize._align_words 消费完一个字后会自动跳过原文中的标点。两者叠加会让
标点被消费两次，严格对齐必然失败并**静默降级**成 jieba 比例插值。
所以这里先滤掉纯标点 token，交给 _align_words 自己吸附。
"""
from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path

from . import config
from .tokenize import _align_words, is_punct_char

MODEL_DIR = config.BASE_DIR / ".runtime" / "models" / "Fun-ASR-Nano-2512"
CODE_DIR = config.BASE_DIR / ".runtime" / "fun_asr_repo"

SENT_END = "。！？!?…"
TRAIL = "。！？!?…\"'”’）)】》"

_MODEL = None
_INIT_LOCK = threading.Lock()   # 模型加载串行
_INFER_LOCK = threading.Lock()  # 推理串行


def available() -> tuple[bool, str]:
    """模型与依赖是否就绪。只查文件，不加载模型。"""
    if not (MODEL_DIR / "model.pt").exists():
        return False, f"模型未下载：{MODEL_DIR}"
    try:
        import funasr  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"依赖缺失：{exc}"
    return True, "ok"


def is_loaded() -> bool:
    return _MODEL is not None


def _load():
    """加载并缓存模型；并发调用只会真正加载一次。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _INIT_LOCK:
        if _MODEL is not None:
            return _MODEL
        import torch
        from funasr import AutoModel

        kwargs = dict(
            model=str(MODEL_DIR),
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            # 不配 punc_model：Fun-ASR-Nano 的文本自带标点，切句用的是 CTC token，
            # 用不上 ct-punc；省掉它能少读约 1.1GB 权重（加载更快）。
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            disable_update=True,
        )
        if (CODE_DIR / "model.py").exists():
            # model.py 里是 `from ctc import CTC` / `tools.utils` 这类绝对导入
            if str(CODE_DIR) not in sys.path:
                sys.path.insert(0, str(CODE_DIR))
            kwargs["remote_code"] = str(CODE_DIR / "model.py")
        _MODEL = AutoModel(**kwargs)
        return _MODEL


def unload() -> tuple[bool, str]:
    """卸载模型并释放显存。正在识别时拒绝，避免推理中途被打断而崩掉。"""
    global _MODEL
    if _MODEL is None:
        return False, "模型未加载"
    if not _INFER_LOCK.acquire(blocking=False):
        return False, "正在识别中，无法释放"
    try:
        _MODEL = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - 释放失败不影响主流程
            pass
        return True, "已释放显存"
    finally:
        _INFER_LOCK.release()


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """按句末标点切分，返回 [(char_start, char_end), ...]，保留标点。"""
    out: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in SENT_END:
            j = i + 1
            while j < n and text[j] in TRAIL:
                j += 1
            out.append((start, j))
            while j < n and text[j].isspace():
                j += 1
            start = j
            i = j
        else:
            i += 1
    if start < n:
        out.append((start, n))
    return [(s, e) for s, e in out if text[s:e].strip()]


def _to_segments(item: dict) -> list[dict]:
    """把模型返回转成项目约定的 segments 结构。"""
    text = (item.get("text") or "").replace("/sil", " ").strip()
    if not text:
        return []

    words: list[dict] = []
    for t in item.get("timestamps") or []:
        tok = (t.get("token") or "").strip()
        if not tok:
            continue
        # 滤掉纯标点：_align_words 会把它吸到前一个词上
        if all(is_punct_char(c) for c in tok):
            continue
        words.append({
            "text": tok,
            "begin": int(round(float(t.get("start_time") or 0) * 1000)),
            "end": int(round(float(t.get("end_time") or 0) * 1000)),
        })

    units = _align_words(text, words) if words else None
    if not units:
        # 兜底：整段交给断句引擎。tokenize 会降级成 jieba + 比例插值，
        # 时间轴会粗糙，但至少不丢结果。
        begin = words[0]["begin"] if words else 0
        end = words[-1]["end"] if words else 0
        if end <= begin:
            last = item.get("timestamps") or []
            if last:
                end = int(round(float(last[-1].get("end_time") or 0) * 1000))
        return [{"begin": begin, "end": max(end, begin + 1),
                 "text": text, "words": words}]

    segs: list[dict] = []
    ui = 0
    for cs, ce in _split_sentences(text):
        seg_units = []
        while ui < len(units) and units[ui].start < ce:
            seg_units.append(units[ui])
            ui += 1
        if not seg_units:
            continue
        segs.append({
            "begin": seg_units[0].begin,
            "end": seg_units[-1].end,
            "text": text[cs:ce].strip(),
            "words": [{"text": text[u.start:u.stop],
                       "begin": u.begin, "end": u.end} for u in seg_units],
        })
    return segs


def transcribe(file_path: str | Path, progress_cb=None) -> tuple[list[dict], dict]:
    """识别一个音频文件，返回 (segments, timing)。

    timing 把两段耗时分开，互不混淆：
      load_sec   本次为加载模型花的时间（复用已加载模型时为 0）
      infer_sec  纯推理耗时（不含加载）
    """
    was_loaded = _MODEL is not None
    t_load = time.time()
    model = _load()
    load_sec = 0.0 if was_loaded else time.time() - t_load

    if progress_cb:
        progress_cb("transcribing")

    t_infer = time.time()
    with _INFER_LOCK:
        res = model.generate(
            input=[str(file_path)],
            cache={},
            batch_size=1,
            language="中文",
            itn=True,
            sentence_timestamp=True,
        )
    infer_sec = time.time() - t_infer

    segments = _to_segments(res[0]) if res and res[0] else []
    return segments, {"load_sec": round(load_sec, 2), "infer_sec": round(infer_sec, 2)}
