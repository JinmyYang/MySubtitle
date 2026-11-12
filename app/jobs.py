"""任务管理：识别任务（转写并保存分段）+ 字幕导出任务（按场景档位断句）。

识别与导出分离：音频只转写一次，分段（含词级时间戳）落盘保存；
之后可按四个档位反复导出字幕，纯本地计算、零 API 成本。

任务本身是**临时**的：不落盘、不跨重启恢复。真正持久的是资产库（assets/）——
识别结果与字幕在生成时即登记为资产，服务重启只重建资产索引，不恢复任务。
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import assets as assets_mod, config, dashscope_client, export_split, srt

# 阶段 -> (进度百分比, 中文描述)
STAGES = {
    "queued": (0, "排队中"),
    "oss_uploading": (15, "正在上传音频到百炼临时存储"),
    "submitted": (25, "已提交转写任务"),
    "PENDING": (35, "服务端排队中"),
    "RUNNING": (60, "模型转写中"),
    "loading_model": (10, "正在加载本地模型（首次约 1 分钟）"),
    "transcribing": (60, "本地模型识别中"),
    "done": (100, "识别完成，可导出字幕"),
}

# 并发上限：同时跑的识别/导出任务数（避免 1GB 上传各占一个裸线程）
_MAX_WORKERS = 4

# 内存里最多保留多少个"已完结"任务。
# 从资产库反复生成字幕会不断新建任务，而任务持有整段识别结果（长音频是 MB 级），
# 不回收的话长会话下内存与 /api/jobs 响应会持续膨胀。真正的结果都在 assets/ 里，
# 这里剪掉的只是内存记录，不影响资产与文件。
_MAX_JOBS = 30


@dataclass
class Export:
    """一次字幕导出：按场景档位生成字幕。"""
    id: str
    job_id: str
    preset: str                # punct/default/frequent/immersive
    status: str = "pending"  # pending/done/failed
    error: str = ""
    srt_path: Path | None = None
    txt_path: Path | None = None
    cue_count: int = 0
    stats: dict = field(default_factory=dict)  # 超宽等排版统计
    preview: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    asset_id: str = ""         # 入库后的字幕资产 id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "preset": self.preset,
            "status": self.status,
            "error": self.error,
            "cue_count": self.cue_count,
            "stats": self.stats,
            "preview": self.preview,
            "created_at": self.created_at,
            "asset_id": self.asset_id,
        }


@dataclass
class Job:
    id: str
    filename: str
    file_path: Path | None      # 上传暂存位置；入库为资产后即置 None
    model: str
    language: str
    engine: str = "local"       # local / cloud，决定 run_pipeline 走哪条链路
    status: str = "queued"  # queued/oss_uploading/submitted/loading_model/transcribing/done/failed
    stage: str = "queued"
    progress: int = 0
    error: str = ""
    task_id: str = ""
    segments: list | None = None       # 识别完成后的句级+词级分段
    detected_lang: str = "zh"          # 固定普通话
    exports: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    duration_sec: float = 0.0          # 音频时长（秒）
    elapsed_sec: float = 0.0           # 本次处理耗时（秒）；导入的任务为 0
    load_sec: float = 0.0              # 模型加载耗时（秒），仅本地引擎；与识别耗时分开记
    # 关联的资产（音频 -> 中间产物 -> 字幕 的链条）
    audio_asset_id: str = ""
    segments_asset_id: str = ""
    segments_file: Path | None = None  # 内存里没有分段时的落盘位置

    def to_dict(self, with_preview: bool = False) -> dict:
        d = {
            "id": self.id,
            "filename": self.filename,
            "model": self.model,
            "engine": self.engine,
            "model_name": ("导入的识别结果" if self.model == "imported" else
                            config.MODELS.get(self.model, {}).get("name", self.model)),
            "language": self.language,
            "detected_lang": self.detected_lang,
            "status": self.status,
            "stage": self.stage,
            "stage_text": STAGES.get(self.stage, (0, self.stage))[1],
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "duration_sec": round(self.duration_sec, 1),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "load_sec": round(self.load_sec, 1),
            "export_count": len(self.exports),
            "audio_asset_id": self.audio_asset_id,
            "segments_asset_id": self.segments_asset_id,
        }
        if with_preview and self.segments:
            # 识别预览：前几句原文（未断句）
            d["preview"] = [
                {"begin": s["begin"], "end": s["end"], "text": s["text"]}
                for s in self.segments[:5]
            ]
        return d

    def meta(self) -> dict:
        return {
            "filename": self.filename,
            "model": self.model,
            "language": self.language,
            "created_at": self.created_at,
            "detected_lang": self.detected_lang,
            "duration_sec": self.duration_sec,
        }


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="job")
# 资产库：音频 / 识别中间产物 / 字幕 都登记在这里，供列表复用
STORE = assets_mod.AssetStore(config.ASSET_DIR)


def create_job(file_path: Path, filename: str, model: str, language: str,
               audio_asset_id: str = "", engine: str = "") -> Job:
    job = Job(
        id=uuid.uuid4().hex[:12],
        filename=filename,
        file_path=file_path,
        model=model,
        language=language,
        engine=engine or config.default_engine(),
        audio_asset_id=audio_asset_id,
    )
    with _LOCK:
        _JOBS[job.id] = job
    _prune_jobs()
    return job


def create_segments_job(asset: assets_mod.Asset, segments: list,
                        model: str) -> Job:
    """从已登记的识别中间产物创建"已完成"任务，用于反复导出不同断句的字幕。"""
    job = Job(
        id=uuid.uuid4().hex[:12],
        # 资产名带"（识别结果）"后缀是为了列表里好认，但用它拼字幕文件名会又长又重复
        filename=asset.name.replace("（识别结果）", ""),
        file_path=None,
        model=model,
        language="zh",
        segments_asset_id=asset.id,
        segments_file=STORE.path_of(asset),
    )
    job.segments = segments
    job.detected_lang = "zh"
    job.status = "done"
    job.stage = "done"
    job.progress = 100
    _set_audio_duration(job, segments)
    with _LOCK:
        _JOBS[job.id] = job
    _prune_jobs()
    return job


def create_imported_job(filename: str, segments: list) -> Job:
    """导入外部识别结果：登记为中间产物资产，再据此创建已完成任务。"""
    dur = max(0.0, float(segments[-1].get("end", 0)) / 1000.0) if segments else 0.0
    # 仅支持普通话识别结果，统一按中文规则导出。
    asset = STORE.add_segments(segments, filename, "zh", "导入", duration_sec=dur)
    return create_segments_job(asset, segments, "imported")


def get_job(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def list_jobs() -> list[Job]:
    return sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)


def _prune_jobs() -> None:
    """回收最老的已完结任务，把内存里的任务数压在 _MAX_JOBS 以内。

    只剪"状态为 done/failed 且所有导出都已结束"的任务，在跑的一律不动。
    结果都已落进 assets/，剪掉内存记录不丢数据。
    必须在把新任务放进 _JOBS **之后**调用，否则会多留一个（新任务占位）。
    """
    with _LOCK:
        excess = len(_JOBS) - _MAX_JOBS
        if excess <= 0:
            return
        finished = [j for j in _JOBS.values()
                    if j.status in ("done", "failed")
                    and all(e.status != "pending" for e in j.exports)]
        finished.sort(key=lambda j: j.created_at)
        for j in finished[:excess]:
            _JOBS.pop(j.id, None)


def _set_stage(job: Job, stage: str) -> None:
    """推进阶段；进度只增不减（服务端状态偶尔回退时进度条不能倒缩）。"""
    job.stage = stage
    pct = STAGES.get(stage, (None, ""))[0]
    if pct is not None:
        job.progress = max(job.progress, pct)


def _set_audio_duration(job: Job, segments: list) -> None:
    if segments:
        job.duration_sec = max(0.0, float(segments[-1].get("end", 0)) / 1000.0)


def _save_segments(job: Job, elapsed_sec: float = 0.0,
                   load_sec: float = 0.0) -> assets_mod.Asset | None:
    """把识别结果登记为"中间产物"资产（可反复导出、可复用）。"""
    if not job.segments:
        return None
    asset = STORE.add_segments(
        job.segments, job.filename, job.detected_lang or job.language,
        job.model, source_id=job.audio_asset_id,
        duration_sec=job.duration_sec, elapsed_sec=elapsed_sec,
        load_sec=load_sec)
    job.segments_asset_id = asset.id
    job.segments_file = STORE.path_of(asset)
    return asset


def run_pipeline(job: Job) -> None:
    """识别流水线：按 job.engine 分发到本地或云端，产出统一的 segments 资产。

    音频资产在上传时就已经入库，识别不再改动它，失败重试用的是同一份文件。
    """
    started = time.time()
    try:
        if job.engine == "local":
            _pipeline_local(job, started)
        else:
            _pipeline_cloud(job, started)
        job.status = "done"
        _set_stage(job, "done")
    except Exception as exc:  # noqa: BLE001 - 任务失败需要完整信息
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.elapsed_sec = time.time() - started


def _pipeline_cloud(job: Job, started: float) -> None:
    """云端：上传音频 -> 提交异步任务 -> 轮询 -> 下载结果。"""
    _set_stage(job, "oss_uploading")
    file_url = dashscope_client.upload_file(job.model, job.file_path)

    _set_stage(job, "submitted")
    job.task_id = dashscope_client.submit_transcription(job.model, file_url)

    job.status = "transcribing"

    def on_poll(task_status: str) -> None:
        _set_stage(job, task_status or "RUNNING")

    output = dashscope_client.wait_for_task(job.task_id, progress_cb=on_poll)
    segments = dashscope_client.fetch_segments(output)
    if not segments:
        raise RuntimeError("转写结果为空")

    job.segments = segments
    job.detected_lang = "zh"
    _set_audio_duration(job, segments)
    # 识别耗时要在入库前落定：存盘、写索引的时间不该算进识别耗时
    elapsed = time.time() - started
    job.elapsed_sec = elapsed
    _save_segments(job, elapsed_sec=elapsed)


def _pipeline_local(job: Job, started: float) -> None:
    """本地：模型按需加载，加载耗时单独记录，不混进识别耗时。"""
    from . import asr_local

    job.status = "transcribing"
    _set_stage(job, "transcribing" if asr_local.is_loaded() else "loading_model")
    segments, timing = asr_local.transcribe(
        job.file_path, progress_cb=lambda s: _set_stage(job, s))
    if not segments:
        raise RuntimeError("本地识别结果为空")

    job.segments = segments
    job.detected_lang = "zh"
    _set_audio_duration(job, segments)
    # elapsed_sec 只放纯推理时间；模型加载是环境成本，单独存 load_sec
    job.elapsed_sec = timing["infer_sec"]
    job.load_sec = timing["load_sec"]
    _save_segments(job, elapsed_sec=timing["infer_sec"],
                   load_sec=timing["load_sec"])


def start_job(job: Job) -> None:
    _EXECUTOR.submit(run_pipeline, job)


# ---------------- 字幕导出 ----------------

def create_export(job: Job, preset: str) -> Export:
    export = Export(
        id=uuid.uuid4().hex[:12],
        job_id=job.id,
        preset=preset,
    )
    job.exports.append(export)
    return export


def run_export(job: Job, export: Export) -> None:
    """导出流水线：按场景档位生成字幕 -> 写 SRT/TXT。"""
    try:
        stats: dict = {}
        cues = export_split.split_segments_by_preset(
            job.segments, export.preset, stats=stats)
        export.stats = stats
        if stats.get("over_limit"):
            print(f"[export] job={job.id} preset={export.preset} "
                  f"超宽独占={stats.get('over_limit', 0)}", flush=True)
        stem = Path(job.filename).stem
        preset_tag = {"punct": "按标点", "default": "默认", "frequent": "频繁",
                      "immersive": "沉浸"}.get(export.preset, export.preset)
        # 先写到系统临时目录，再由资产库收编（收编时会移动到 assets/subtitles/）。
        # 不再借用 results/ 当暂存区——文件被移走后目录会留空，导出几百次就是几百个空目录。
        with tempfile.TemporaryDirectory(prefix="subtitle_") as tmp_dir:
            srt_tmp = Path(tmp_dir) / f"{stem}_{preset_tag}.srt"
            txt_tmp = Path(tmp_dir) / f"{stem}_{preset_tag}.txt"
            srt_tmp.write_text(srt.segments_to_srt(cues), encoding="utf-8")
            txt_tmp.write_text(srt.segments_to_txt(cues), encoding="utf-8")
            export.cue_count = len(cues)
            export.preview = [
                {"begin": c["begin"], "end": c["end"], "text": c["text"]}
                for c in cues[:5]
            ]
            asset = STORE.add_subtitle(
                srt_tmp, txt_tmp, srt_tmp.name, job.detected_lang, job.model,
                export.preset, len(cues),
                source_id=job.segments_asset_id, duration_sec=job.duration_sec)
        export.asset_id = asset.id
        export.srt_path = STORE.path_of(asset)
        export.txt_path = export.srt_path.with_suffix(".txt")
        export.status = "done"
    except Exception as exc:  # noqa: BLE001
        export.status = "failed"
        export.error = str(exc)


def start_export(job: Job, export: Export) -> None:
    _EXECUTOR.submit(run_export, job, export)


# ---------------- 启动恢复 ----------------

def load_assets() -> int:
    """启动时重建资产索引。任务本身是临时的，资产才是持久的。"""
    try:
        return STORE.load()
    except Exception as exc:  # noqa: BLE001 - 索引损坏不能阻止启动
        print(f"[startup] 资产索引加载失败（已忽略）: {exc}", flush=True)
        return 0


def load_segments(asset_id: str) -> list[dict]:
    """按资产 id 读取识别中间产物。"""
    asset = STORE.get(asset_id)
    if asset is None or asset.kind != assets_mod.KIND_SEGMENTS:
        return []
    return json.loads(STORE.path_of(asset).read_text(encoding="utf-8"))
