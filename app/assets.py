"""资产管理：把上传的音频、识别中间产物、生成的字幕统一登记为可复用的一等公民。

设计要点：
- 三类资产（audio / segments / subtitle）共用一套元数据模型与索引，
  前端用一个列表就能按类型切换展示。
- 每个资产都记录"由来"：语言、使用的模型；字幕额外记录断句档位。
  这样隔天回来看列表，也知道每条字幕是怎么来的。
- 资产之间用 source_id 串成链：音频 -> 中间产物 -> 字幕。
  字幕预览时顺着链往上找就能拿到对应的音频，做音画同步预览。
- 索引落盘为 index.json，服务重启后重建（文件本身才是权威来源）。

目录布局（assets/ 下，运行期产物，不进版本库）：
    assets/audio/{id}_{filename}
    assets/segments/{id}_{stem}.json
    assets/subtitles/{id}_{stem}.srt  与同名 .txt
    assets/index.json
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

KIND_AUDIO = "audio"
KIND_SEGMENTS = "segments"
KIND_SUBTITLE = "subtitle"
KINDS = (KIND_AUDIO, KIND_SEGMENTS, KIND_SUBTITLE)

_INDEX_NAME = "index.json"


def safe_name(name: str) -> str:
    """清理文件名，避免路径穿越与非法字符。"""
    name = unicodedata.normalize("NFC", name or "unnamed").replace("\\", "/").split("/")[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    return name[:120] or "unnamed"


@dataclass
class Asset:
    id: str
    kind: str                 # audio / segments / subtitle
    name: str                 # 显示名
    file: str                 # 相对 assets/ 的路径
    created_at: float = field(default_factory=time.time)
    # —— 由来 ——
    language: str = ""        # 固定普通话；保留字段兼容旧索引
    model: str = ""           # 识别模型，或 imported
    # —— 字幕专有 ——
    split_mode: str = ""      # 断句档位（沿用字段名兼容旧索引）
    cue_count: int = 0
    # —— 其它 ——
    duration_sec: float = 0.0
    size_bytes: int = 0
    source_id: str = ""       # 由哪个资产生成
    note: str = ""            # 备注：索引重建时说明"由来未知"等

    def to_dict(self) -> dict:
        d = asdict(self)
        d["origin"] = self.origin_text()
        return d

    def origin_text(self) -> str:
        """人类可读的"由来"描述，前端直接展示。"""
        bits = ["普通话"]
        if self.model:
            bits.append(self.model)
        if self.kind == KIND_SUBTITLE and self.split_mode:
            mode = {"punct": "按标点", "default": "默认", "frequent": "频繁",
                    "immersive": "沉浸", "rule": "旧版断句",
                    "llm": "旧版断句"}.get(self.split_mode, self.split_mode)
            bits.append(mode)
        if self.note:
            bits.append(self.note)      # 例如索引重建时补的"由来未知"
        return " · ".join(bits) if bits else "—"


class AssetStore:
    """资产存储：文件放在 assets/ 下，元数据集中在 index.json。"""

    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / _INDEX_NAME
        self._assets: dict[str, Asset] = {}
        self._lock = threading.Lock()

    # ---- 持久化 ----

    def load(self) -> int:
        """启动时重建索引：先读 index.json，再扫目录补上索引里没有的文件。

        index.json 只是缓存，**文件本身才是权威来源**。索引损坏、被删，或有人手工把
        文件拷回 assets/ 时，只信索引会让资产凭空消失；补一遍目录扫描就能捞回来。
        """
        self._load_index()
        adopted = self._adopt_orphan_files()
        if adopted:
            self.save()
        return len(self._assets)

    def _load_index(self) -> None:
        """读 index.json；顺手丢掉文件已不存在的记录。"""
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 索引损坏就交给目录扫描兜底
            return
        with self._lock:
            for item in raw if isinstance(raw, list) else []:
                try:
                    a = Asset(**{k: v for k, v in item.items()
                                 if k in Asset.__dataclass_fields__})
                except TypeError:
                    continue
                if (self.root / a.file).exists():
                    self._assets[a.id] = a

    def _adopt_orphan_files(self) -> int:
        """扫描 assets/ 各子目录，把索引里没有的文件登记进来。

        文件名固定为 `{id}_{原始名}`（见 add_* 方法），据此还原 id 与显示名。
        语言、模型、断句档位这些"由来"信息文件本身不携带，只能标注为未知。
        """
        adopted = 0
        for kind in KINDS:
            folder = self.root / kind
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file() or self._is_sibling_file(kind, path):
                    continue
                stem = path.name if kind == KIND_AUDIO else path.name.rsplit(".", 1)[0]
                asset_id, _, origin = stem.partition("_")
                if len(asset_id) != 12 or not origin:
                    continue          # 不是本工具生成的命名，不碰
                with self._lock:
                    if asset_id in self._assets:
                        continue
                a = Asset(
                    id=asset_id, kind=kind,
                    name=path.name if kind == KIND_AUDIO else f"{origin}{path.suffix}",
                    file=f"{kind}/{path.name}",
                    created_at=path.stat().st_mtime,
                    size_bytes=path.stat().st_size,
                    note="索引重建，由来未知",
                )
                if kind == KIND_SEGMENTS:
                    self._probe_segments(a, path)
                with self._lock:
                    self._assets[a.id] = a
                adopted += 1
        return adopted

    @staticmethod
    def _is_sibling_file(kind: str, path: Path) -> bool:
        """同伴文件不算独立资产：字幕的 .txt 跟着 .srt 走。"""
        if kind == KIND_SUBTITLE:
            return path.suffix.lower() != ".srt"
        if kind == KIND_SEGMENTS:
            return path.suffix.lower() != ".json"
        return False

    @staticmethod
    def _probe_segments(asset: Asset, path: Path) -> None:
        """从识别结果文件里量出条数与时长（失败就不填，不影响登记）。"""
        try:
            segs = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(segs, list) and segs:
                asset.cue_count = len(segs)
                last = segs[-1].get("end")
                if isinstance(last, (int, float)):
                    asset.duration_sec = max(0.0, float(last) / 1000.0)
        except Exception:  # noqa: BLE001 - 读不动就保持未知
            pass

    def retarget(self, root: Path) -> None:
        """切换存储根目录（测试隔离用）；index_path 必须跟着走。"""
        self.root = Path(root)
        self.index_path = self.root / _INDEX_NAME
        with self._lock:
            self._assets.clear()

    def save(self) -> None:
        """落盘索引。

        整段（含临时文件与 replace）都在锁内，且临时文件名唯一——
        否则并发导出时多个线程会抢同一个 index.tmp，Windows 上直接
        WinError 32「另一个程序正在使用此文件」。
        """
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(a) for a in sorted(self._assets.values(),
                                              key=lambda x: x.created_at, reverse=True)]
            tmp = self.index_path.with_name(
                f"{_INDEX_NAME}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                tmp.replace(self.index_path)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    # ---- 查询 ----

    def list(self, kind: str | None = None) -> list[Asset]:
        with self._lock:
            items = list(self._assets.values())
        if kind:
            items = [a for a in items if a.kind == kind]
        return sorted(items, key=lambda a: a.created_at, reverse=True)

    def get(self, asset_id: str) -> Asset | None:
        with self._lock:
            return self._assets.get(asset_id)

    def path_of(self, asset: Asset) -> Path:
        return self.root / asset.file

    def remove(self, asset_id: str, cascade: bool = False) -> list[str] | None:
        """删除资产及其文件；cascade 时连同由它派生的资产一起删。

        返回被删除的 id 列表；资产不存在时返回 None。
        不级联删除会留下 source_id 悬空的派生资产（字幕预览再也找不到音频），
        所以前端会先告知派生数量，让用户决定是否一并清理。
        """
        with self._lock:
            if asset_id not in self._assets:
                return None
            ids = [asset_id] + (self._descendants_locked(asset_id) if cascade else [])
            victims = [self._assets.pop(i) for i in ids]
        for a in victims:
            self._unlink(a)
        self.save()
        return ids

    def _descendants_locked(self, asset_id: str) -> list[str]:
        """按 source_id 反查派生资产（调用方需已持锁）。"""
        out: list[str] = []
        frontier = [asset_id]
        while frontier:
            cur = frontier.pop()
            for a in self._assets.values():
                if a.source_id == cur and a.id not in out:
                    out.append(a.id)
                    frontier.append(a.id)
        return out

    def derived_count(self, asset_id: str) -> int:
        """由该资产派生出来的资产数量。"""
        with self._lock:
            return len(self._descendants_locked(asset_id))

    def describe(self, asset: Asset) -> dict:
        """资产字典（附加 derived_count，供前端删除前提示）。"""
        d = asset.to_dict()
        d["derived_count"] = self.derived_count(asset.id)
        return d

    def _unlink(self, a: Asset) -> None:
        """删除资产对应的文件；字幕连带删掉同名 .txt。"""
        p = self.root / a.file
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        if a.kind == KIND_SUBTITLE:
            for suffix in (".srt", ".txt"):
                sib = p.with_suffix(suffix)
                if sib.is_file():
                    sib.unlink(missing_ok=True)

    # ---- 登记 ----

    def _register(self, asset: Asset) -> Asset:
        with self._lock:
            self._assets[asset.id] = asset
        self.save()
        return asset

    def add_audio(self, src: Path, filename: str, language: str = "auto",
                  model: str = "") -> Asset:
        """把上传的音频收进资产库（移动，不保留 uploads/ 里的副本）。"""
        uid = uuid.uuid4().hex[:12]
        name = safe_name(filename)
        rel = f"{KIND_AUDIO}/{uid}_{name}"
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return self._register(Asset(
            id=uid, kind=KIND_AUDIO, name=name, file=rel,
            language=language, model=model,
            size_bytes=dst.stat().st_size,
        ))

    def add_segments(self, segments: list[dict], name: str, language: str,
                     model: str, source_id: str = "",
                     duration_sec: float = 0.0) -> Asset:
        uid = uuid.uuid4().hex[:12]
        stem = safe_name(Path(name).stem or name)
        rel = f"{KIND_SEGMENTS}/{uid}_{stem}.json"
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        return self._register(Asset(
            id=uid, kind=KIND_SEGMENTS, name=f"{stem}（识别结果）", file=rel,
            language=language, model=model, source_id=source_id,
            duration_sec=duration_sec, size_bytes=dst.stat().st_size,
            cue_count=len(segments),
        ))

    def add_subtitle(self, srt_src: Path, txt_src: Path, name: str,
                     language: str, model: str, preset: str,
                     cue_count: int, source_id: str,
                     duration_sec: float = 0.0) -> Asset:
        uid = uuid.uuid4().hex[:12]
        stem = safe_name(Path(name).stem or name)
        rel_srt = f"{KIND_SUBTITLE}/{uid}_{stem}.srt"
        rel_txt = f"{KIND_SUBTITLE}/{uid}_{stem}.txt"
        (self.root / KIND_SUBTITLE).mkdir(parents=True, exist_ok=True)
        shutil.move(str(srt_src), str(self.root / rel_srt))
        shutil.move(str(txt_src), str(self.root / rel_txt))
        return self._register(Asset(
            id=uid, kind=KIND_SUBTITLE, name=f"{stem}.srt", file=rel_srt,
            language=language, model=model, split_mode=preset,
            cue_count=cue_count, source_id=source_id,
            duration_sec=duration_sec,
            size_bytes=(self.root / rel_srt).stat().st_size,
        ))

    def chain_audio_id(self, asset: Asset) -> str | None:
        """顺着 字幕 -> 中间产物 -> 音频 的链往上找音频资产。"""
        cur = asset
        for _ in range(4):
            if cur.kind == KIND_AUDIO:
                return cur.id
            if not cur.source_id:
                return None
            nxt = self.get(cur.source_id)
            if nxt is None:
                return None
            cur = nxt
        return None
