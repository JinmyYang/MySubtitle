"""FastAPI 应用：上传音频 -> 百炼离线转写 -> 按场景档位多次导出字幕。

启动：`python scripts/serve.py`（自动选可用端口），或 `uvicorn app.main:app --port 18080`。
默认端口是 **18080**，不是 8000：同机 Applio Desktop 硬编码占用 8000（API）/ 3000（界面），
本项目不要再回到那两个端口。端口也不写死——Windows 会动态保留一段 TCP 端口，绑不上会自动后移。

三处容易被忽略但很关键的取舍：
1. 上传与导入接口刻意用同步 def。大文件读写是同步 I/O，放在 async 路由里会
   堵死事件循环（前端每 2 秒一次的轮询会全部超时）；同步路由会被 FastAPI 丢进线程池。
2. 写盘成功之后才登记任务，避免磁盘写失败时留下一个永远停在 queued 的僵尸任务。
3. 中间件先卡 Host 再卡 Origin。只比 Origin 与 Host 挡不住 DNS rebinding——
   域名解析到 127.0.0.1 后两者会同时变成攻击者的域名、天然相等。
"""
from __future__ import annotations

import ipaddress
import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import assets as assets_mod, asr_local, config, jobs, srt


def _cleanup_stale_uploads() -> int:
    """清掉上次运行中断留下的上传临时文件。

    客户端中途断开时（starlette 抛 ClientDisconnect）或进程被杀，落盘到一半的
    `.tmp_*` 就留在 uploads/ 了；启动时统一收拾一次，避免无限累积。
    """
    if not config.UPLOAD_DIR.exists():
        return 0
    removed = 0
    for p in config.UPLOAD_DIR.glob(".tmp_*"):
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except OSError:
            pass          # 正被别的进程占用就留到下次
    return removed


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时重建资产索引（音频 / 中间产物 / 字幕），任务本身是临时的。"""
    try:
        n = jobs.load_assets()
        if n:
            print(f"[startup] 已从 assets/ 载入 {n} 项资产", flush=True)
    except Exception as exc:  # noqa: BLE001 - 加载失败不能阻止启动
        print(f"[startup] 资产载入失败（已忽略）: {exc}", flush=True)
    try:
        stale = _cleanup_stale_uploads()
        if stale:
            print(f"[startup] 已清理 {stale} 个残留的上传临时文件", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] 上传临时文件清理失败（已忽略）: {exc}", flush=True)
    yield


app = FastAPI(title="音频转字幕", version="1.0", lifespan=lifespan)

_CHUNK = 8 * 1024 * 1024   # 上传落盘的分块大小


def _is_local_host(host_header: str) -> bool:
    """Host 是否指向本机（回环地址）。

    只比 Origin 与 Host 是挡不住 DNS rebinding 的：把域名解析到 127.0.0.1 后，
    浏览器发出的 Host 与 Origin 会**同时**变成攻击者的域名，两者天然相等。
    所以必须先确认请求确实是冲着本机来的，再谈同源。
    """
    host = (host_header or "").strip()
    if not host:
        return False
    if host.startswith("["):                      # IPv6 字面量：[::1]:18080
        host = host[1:host.index("]")] if "]" in host else host[1:]
    elif ":" in host:                             # IPv4 / 主机名带端口
        host = host.rsplit(":", 1)[0]
    host = host.lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def _local_only_guard(request: Request, call_next):
    """只服务本机，并拒绝跨站写请求。

    服务只监听 127.0.0.1，但浏览器里的任意页面都能向它发请求，所以两道都要做：
      1. Host 必须是回环地址——挡住 DNS rebinding（读接口同样要挡，否则音频/字幕会被读走）；
      2. 写操作 Origin 必须与 Host 同源——挡住 CSRF。
    """
    if not _is_local_host(request.headers.get("host", "")):
        return JSONResponse({"detail": "只接受来自本机的请求"}, status_code=403)
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        origin = request.headers.get("origin", "")
        if origin and urlparse(origin).netloc != request.headers.get("host", ""):
            return JSONResponse({"detail": "跨站请求被拒绝"}, status_code=403)
    return await call_next(request)


@app.get("/api/models")
def api_models():
    local_ok, local_hint = asr_local.available()
    return {
        "models": [
            {"code": code, "name": spec["name"], "default": spec.get("default", False)}
            for code, spec in config.MODELS.items()
        ],
        "default_model": config.default_model(),
        "engines": [
            {"code": code, "name": spec["name"], "default": spec.get("default", False)}
            for code, spec in config.ENGINES.items()
        ],
        "default_engine": config.default_engine(),
        "local_ready": local_ok,
        "local_hint": local_hint,
        "local_loaded": asr_local.is_loaded(),
        "presets": config.SPLIT_PRESETS,
        "default_preset": config.DEFAULT_PRESET,
    }


@app.get("/api/asr/status")
def api_asr_status():
    """本地模型状态：页面用来决定「释放显存」按钮是否可点。"""
    ok, hint = asr_local.available()
    return {"ready": ok, "hint": hint, "loaded": asr_local.is_loaded()}


@app.post("/api/asr/unload")
def api_asr_unload():
    """卸载本地模型、释放显存。正在识别时拒绝，避免打断推理。"""
    ok, msg = asr_local.unload()
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg, "loaded": asr_local.is_loaded()}


def _resolve_engine(engine: str) -> tuple[str, str]:
    """校验引擎并返回 (engine, 写进资产/任务的模型名)。"""
    engine = engine or config.default_engine()
    if engine not in config.ENGINES:
        raise HTTPException(400, f"不支持的识别引擎: {engine}")
    if engine == "local":
        ok, hint = asr_local.available()
        if not ok:
            raise HTTPException(400, f"本地引擎不可用：{hint}")
        return engine, config.LOCAL_MODEL_TAG
    return engine, config.default_model()


def _safe_name(name: str) -> str:
    """清理上传文件名（与资产库共用同一套规则，避免两份实现各自漂移）。"""
    return assets_mod.safe_name(name)


@app.post("/api/jobs")
def api_create_job(file: UploadFile = File(...), engine: str = Form("")):
    """上传普通话音频并启动识别。同步路由：落盘交给线程池更稳。"""
    engine, model = _resolve_engine(engine)
    filename = _safe_name(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的文件格式: {ext or '无扩展名'}")

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # 先落盘到临时名，写成功后再改成正式名（并交给资产库收编）
    tmp = config.UPLOAD_DIR / f".tmp_{uuid.uuid4().hex[:8]}_{filename}"
    size = 0
    try:
        with tmp.open("wb") as out:
            while chunk := file.file.read(_CHUNK):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "文件超过 1GB 上限")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "上传的文件为空")
        # 音频一上传就进资产库，之后反复识别/导出都复用这一份
        asset = jobs.STORE.add_audio(tmp, filename, "zh", model)
        job = jobs.create_job(file_path=jobs.STORE.path_of(asset),
                              filename=filename, model=model, language="zh",
                              audio_asset_id=asset.id, engine=engine)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"保存上传文件失败: {exc}") from exc
    except BaseException:
        # 客户端中途断开时 starlette 抛的是 ClientDisconnect，不是 OSError；
        # 不兜底的话临时文件会一直留在 uploads/ 里
        tmp.unlink(missing_ok=True)
        raise

    jobs.start_job(job)
    return job.to_dict()


# ---------------- 资产库 ----------------
# 音频 / 识别中间产物 / 字幕 都登记在这里，可挑选后再次发起识别或导出。

@app.get("/api/assets")
def api_list_assets(kind: str = ""):
    """资产列表；kind 为空时返回全部，并按类型给出计数。

    每条附带 derived_count（由它派生的资产数），前端删除前据此提示。
    """
    if kind and kind not in assets_mod.KINDS:
        raise HTTPException(400, f"不支持的资产类型: {kind}")
    items = jobs.STORE.list(kind or None)
    return {
        "assets": [jobs.STORE.describe(a) for a in items],
        "counts": {k: len(jobs.STORE.list(k)) for k in assets_mod.KINDS},
    }


@app.post("/api/assets/{asset_id}/transcribe")
def api_transcribe_asset(asset_id: str, engine: str = Form("")):
    """拿一份已保存的普通话音频重新识别。"""
    asset = jobs.STORE.get(asset_id)
    if asset is None or asset.kind != assets_mod.KIND_AUDIO:
        raise HTTPException(404, "音频资产不存在")
    engine, model = _resolve_engine(engine)
    src = jobs.STORE.path_of(asset)
    if not src.exists():
        raise HTTPException(404, "音频文件已丢失")
    job = jobs.create_job(file_path=src, filename=asset.name, model=model,
                          language="zh", audio_asset_id=asset.id, engine=engine)
    jobs.start_job(job)
    return job.to_dict()


@app.post("/api/assets/{asset_id}/export")
def api_export_asset(asset_id: str, preset: str = Form("")):
    """拿一份已保存的识别结果，按场景档位再生成一版字幕。"""
    asset = jobs.STORE.get(asset_id)
    if asset is None or asset.kind != assets_mod.KIND_SEGMENTS:
        raise HTTPException(404, "识别结果资产不存在")
    segments = jobs.load_segments(asset_id)
    if not segments:
        raise HTTPException(400, "识别结果为空或已损坏")
    preset = preset or config.DEFAULT_PRESET
    if preset not in {p["code"] for p in config.SPLIT_PRESETS}:
        raise HTTPException(400, f"不支持的断句档位: {preset}")
    job = jobs.create_segments_job(asset, segments, asset.model)
    export = jobs.create_export(job, preset)
    jobs.start_export(job, export)
    return export.to_dict()


@app.get("/api/assets/{asset_id}/preview")
def api_preview_asset(asset_id: str):
    """字幕预览：解析成条目，并顺资产链找到对应音频，便于网页同步预览。"""
    asset = jobs.STORE.get(asset_id)
    if asset is None:
        raise HTTPException(404, "资产不存在")
    if asset.kind != assets_mod.KIND_SUBTITLE:
        raise HTTPException(400, "只有字幕可以预览")
    path = jobs.STORE.path_of(asset)
    if not path.exists():
        raise HTTPException(404, "字幕文件已丢失")
    cues = srt.parse_srt(path)
    audio_id = jobs.STORE.chain_audio_id(asset)
    return {
        "id": asset.id,
        "name": asset.name,
        "origin": asset.origin_text(),
        "language": asset.language,
        "cue_count": len(cues),
        "cues": cues,
        "audio_id": audio_id or "",
        "audio_url": f"/api/assets/{audio_id}/file" if audio_id else "",
    }


# "bytes=start-end" / "bytes=start-" / "bytes=-N" 三种形式
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 256 * 1024


@app.get("/api/assets/{asset_id}/file")
def api_asset_file(asset_id: str, request: Request, download: int = 0):
    """读取资产文件本身（音频用于播放，字幕用于下载）。

    必须支持 Range：浏览器只有在响应带 206 / Accept-Ranges 时才会给出 seekable
    区间，否则给它设 currentTime 会被静默忽略——表现为"点字幕不跳转、播放不高亮"。
    当前依赖版本的 FileResponse 不做 Range，所以这里自己实现；
    顺带把 download 参数真正用起来（此前无论传什么都会带上 attachment）。
    """
    asset = jobs.STORE.get(asset_id)
    if asset is None:
        raise HTTPException(404, "资产不存在")
    path = jobs.STORE.path_of(asset)
    if not path.exists():
        raise HTTPException(404, "文件已丢失")

    media = "application/x-subrip" if path.suffix == ".srt" else None
    disp = "attachment" if download else "inline"
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"{disp}; filename*=utf-8''{quote(asset.name)}",
    }
    size = path.stat().st_size

    m = _RANGE_RE.fullmatch(request.headers.get("range", "").strip())
    if m is None or not (m.group(1) or m.group(2)):
        return FileResponse(path, media_type=media, headers=headers)

    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:                                   # "bytes=-N"：末尾 N 字节
        start, end = max(0, size - int(m.group(2))), size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def _chunks():
        remaining = end - start + 1
        with path.open("rb") as f:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(_CHUNK, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(_chunks(), status_code=206, media_type=media, headers=headers)


@app.delete("/api/assets/{asset_id}")
def api_delete_asset(asset_id: str, cascade: int = 0):
    """删除资产；cascade=1 时连同由它派生的资产（识别结果 / 字幕）一起删。"""
    removed = jobs.STORE.remove(asset_id, cascade=bool(cascade))
    if not removed:
        raise HTTPException(404, "资产不存在")
    return {"ok": True, "removed": removed}


@app.get("/api/jobs")
def api_list_jobs():
    return {"jobs": [j.to_dict() for j in jobs.list_jobs()]}


@app.post("/api/jobs/import")
def api_import_segments(file: UploadFile = File(...)):
    """导入识别中间产物（segments.json），生成可直接导出的已完成任务，
    免重复识别，方便反复测试各断句档位。"""
    filename = _safe_name(file.filename)
    if Path(filename).suffix.lower() != ".json":
        raise HTTPException(400, "仅支持 .json 识别结果文件")
    raw = file.file.read(50 * 1024 * 1024 + 1)
    if not raw or len(raw) > 50 * 1024 * 1024:
        raise HTTPException(400, "文件为空或超过 50MB")
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "不是有效的 JSON 文件")
    if not isinstance(data, list) or not data:
        raise HTTPException(400, "识别结果应为非空数组")
    segments = []
    for it in data:
        if (not isinstance(it, dict) or "text" not in it
                or not isinstance(it.get("begin"), (int, float))
                or not isinstance(it.get("end"), (int, float))):
            raise HTTPException(400, "识别结果格式不正确（每段需含 begin/end/text）")
        segments.append({"begin": int(it["begin"]), "end": int(it["end"]),
                         "text": str(it["text"]), "words": it.get("words", [])})
    job = jobs.create_imported_job(Path(filename).stem, segments)
    return job.to_dict(with_preview=True)


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.to_dict(with_preview=True)


@app.post("/api/jobs/{job_id}/exports")
def api_create_export(job_id: str, preset: str = Form("")):
    """对已识别任务按场景档位导出一版字幕（可反复导出）。"""
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status != "done" or not job.segments:
        raise HTTPException(409, f"任务尚未识别完成，当前状态: {job.status}")
    preset = preset or config.DEFAULT_PRESET
    if preset not in {p["code"] for p in config.SPLIT_PRESETS}:
        raise HTTPException(400, f"不支持的断句档位: {preset}")
    export = jobs.create_export(job, preset)
    jobs.start_export(job, export)
    return export.to_dict()


@app.get("/api/jobs/{job_id}/exports")
def api_list_exports(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return {"exports": [e.to_dict() for e in reversed(job.exports)]}


@app.get("/api/jobs/{job_id}/exports/{export_id}/download")
def api_export_download(job_id: str, export_id: str, format: str = "srt"):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    export = next((e for e in job.exports if e.id == export_id), None)
    if not export:
        raise HTTPException(404, "导出记录不存在")
    if export.status != "done":
        raise HTTPException(409, f"导出尚未完成，当前状态: {export.status}")
    if format == "srt":
        path, media = export.srt_path, "application/x-subrip"
    elif format == "txt":
        path, media = export.txt_path, "text/plain"
    else:
        raise HTTPException(400, f"不支持的格式: {format}")
    if not path or not path.exists():
        raise HTTPException(404, "字幕文件不存在")
    return FileResponse(path, media_type=media, filename=path.name)


# 静态页面
app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static"),
                           html=True), name="static")
