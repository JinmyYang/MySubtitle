"""无头端到端测试：识别/导出分离全流程（mock 百炼接口，不耗 API）。

用 pytest 组织：每个用例通过 fixture 拿到干净的全局状态（百炼 mock、任务表），
可以单独重跑、乱序跑、并行跑，不会互相污染。

运行:
    pytest tests/ -v
    python tests/test_api.py        # 等价，内部转发给 pytest
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import (  # noqa: E402
    assets as assets_mod, config, dashscope_client, export_split, jobs, main,
    split_base, split_presets, split_punct, tokenize,
)

ROOT = Path(__file__).resolve().parent.parent

USER_CASE = "那DGX SPARK这颗GP10核心呢，其实是由两颗芯粒组成的，总面积达到了380平方毫米以上。"

# 真实识别结果上的不变量回归。样本目录不进版本库，找到几份跑几份。
SAMPLE_DIRS = (
    ROOT / "samples",
    ROOT / "assets" / "segments",
    ROOT / "results" / "preset_eval" / "segments",
)
BEAT_PRESETS = ("default", "frequent", "immersive")

# mock 的识别结果（含词级时间戳）
FAKE_SEGMENTS = [
    {"begin": 0, "end": 2000, "text": "你好，世界。",
     "words": [{"text": "你好", "begin": 0, "end": 900},
               {"text": "，", "begin": 900, "end": 1000},
               {"text": "世界", "begin": 1000, "end": 1900},
               {"text": "。", "begin": 1900, "end": 2000}]},
    {"begin": 2000, "end": 6000, "text": "这台笔记本电脑的显示器非常清晰",
     "words": [{"text": "这台", "begin": 2000, "end": 2500},
               {"text": "笔记本", "begin": 2500, "end": 3300},
               {"text": "电脑", "begin": 3300, "end": 3800},
               {"text": "的", "begin": 3800, "end": 4000},
               {"text": "显示器", "begin": 4000, "end": 4800},
               {"text": "非常", "begin": 4800, "end": 5300},
               {"text": "清晰", "begin": 5300, "end": 6000}]},
]


# ---------------- 辅助 ----------------

def _nows(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _check_token_bounds(segments: list[dict], cues: list[dict],
                        language: str) -> list[str]:
    """核对每个断点都落在 token 边界上。返回违规片段。

    在“整段去空白”的坐标上比对：一条字幕可能跨越多个 ASR 句（短句合并），
    因此不能逐句消费，必须在全局 token 流上顺序核对。跨句回收的引号也是 token
    的一部分，所以边界必须从 `build_token_stream` 取，不能按单句拼。
    """
    full = _nows("".join(s["text"] for s in segments))
    bounds: set[int] = {0}
    cum = 0
    for t in tokenize.build_token_stream(segments, language):
        cum += len(_nows(t.text))
        bounds.add(cum)
    bounds.add(len(full))

    bad: list[str] = []
    pos = 0
    for c in cues:
        cc = _nows(c["text"])
        if not cc:
            continue
        if not full.startswith(cc, pos):
            return [f"<无法顺序对齐: {full[pos:pos + 12]}> vs {cc[:12]}"]
        pos += len(cc)
        if pos < len(full) and pos not in bounds:
            bad.append(full[max(0, pos - 6):pos + 6])
    return bad


# ---------------- fixtures ----------------

@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """每个用例前重置全局状态与百炼 mock，确保用例之间零污染。"""
    # 资产库与上传目录都指向临时目录，避免测试把文件写进真实的 assets/ 与 uploads/
    jobs.STORE.retarget(tmp_path / "assets")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        dashscope_client, "upload_file",
        lambda model, path, progress_cb=None: f"oss://fake/{model}/{Path(path).name}")
    monkeypatch.setattr(dashscope_client, "submit_transcription",
                        lambda model, url: "fake-task-id")
    monkeypatch.setattr(
        dashscope_client, "wait_for_task",
        lambda task_id, poll_interval=3.0, timeout=3600, progress_cb=None: {"task_status": "SUCCEEDED"})
    monkeypatch.setattr(
        dashscope_client, "fetch_segments",
        lambda output: [dict(s, words=[dict(w) for w in s["words"]]) for s in FAKE_SEGMENTS])
    jobs._JOBS.clear()
    yield
    jobs._JOBS.clear()


@pytest.fixture
def client():
    # 中间件只接受回环 Host，TestClient 默认的 "testserver" 会被拦，显式指成 127.0.0.1
    return TestClient(main.app, base_url="http://127.0.0.1:18080")


@pytest.fixture
def done_job(client):
    """创建一个已识别完成的任务，返回任务详情。"""
    r = client.post("/api/jobs",
                    files={"file": ("测试音频.mp3", io.BytesIO(b"\x00" * 4096))})
    assert r.status_code == 200, r.text
    info = r.json()
    for _ in range(50):
        info = client.get(f"/api/jobs/{info['id']}").json()
        if info["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert info["status"] == "done", f"识别失败: {info.get('error')}"
    assert info["detected_lang"] == "zh", "落地语种应为中文"
    assert len(info["preview"]) == 2, "识别预览为原文前几句"
    return info


@pytest.fixture
def exports(client, done_job):
    """在同一个任务上建好四个档位，返回导出信息。"""
    jid = done_job["id"]
    out = {}
    for preset in ("punct", "default", "frequent", "immersive"):
        r = client.post(f"/api/jobs/{jid}/exports", data={"preset": preset})
        assert r.status_code == 200, r.text
        assert r.json()["preset"] == preset
        out[preset] = r.json()

    done = {}
    for _ in range(100):
        lst = client.get(f"/api/jobs/{jid}/exports").json()["exports"]
        done = {e["id"]: e for e in lst}
        if all(e["status"] != "pending" for e in lst):
            break
        time.sleep(0.1)
    for key in out:
        assert done[out[key]["id"]]["status"] == "done", \
            f"{key} 导出未完成: {done[out[key]['id']]}"
    return {"meta": out, "done": done}


def _discover_real_samples() -> list[Path]:
    """发现普通话识别结果 JSON；外语历史资产不参与回归。"""
    found: list[Path] = []
    for folder in SAMPLE_DIRS:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                segments = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - 不是识别结果就跳过
                continue
            if not (isinstance(segments, list) and segments and "text" in segments[0]):
                continue
            text = "".join(str(s.get("text") or "") for s in segments)
            if sum("\u4e00" <= ch <= "\u9fff" for ch in text) >= 20:
                found.append(path)
    return found


def _real_params():
    return [(path, preset)
            for path in _discover_real_samples()
            for preset in BEAT_PRESETS]


# ---------------- 断句引擎 ----------------

def test_engine_punct():
    """按标点：每个小句一条，不控字数。"""
    seg = {"begin": 0, "end": 20000, "text": USER_CASE, "words": []}
    cues = export_split.split_by_preset([seg], "punct")
    texts = [c["text"] for c in cues]
    assert texts == ["那DGX SPARK这颗GP10核心呢，", "其实是由两颗芯粒组成的，",
                     "总面积达到了380平方毫米以上。"], f"标点直断、每子句一条: {texts}"
    assert _nows("".join(texts)) == _nows(USER_CASE), "文本无损"
    assert all(cues[i]["begin"] <= cues[i + 1]["begin"]
               for i in range(len(cues) - 1)), "时间单调"


@pytest.mark.parametrize("preset", BEAT_PRESETS)
def test_presets_preserve_structure(preset):
    """三个节奏档位都必须满足：无损、不切词、时间单调、不超限。"""
    text = (USER_CASE.rstrip("。") + "，而且它的功耗控制得相当不错，"
                                     "散热设计也很有意思，价格方面嘛，目前还没有正式公布")
    seg = {"begin": 0, "end": 30000, "text": text, "words": []}
    cues = export_split.split_by_preset([seg], preset)
    spec = split_presets.PRESETS[preset]

    assert _nows("".join(c["text"] for c in cues)) == _nows(text), f"{preset}: 文本无损"
    assert all(split_presets.text_width(c["text"]) / 2 <= spec.max_chars + 0.01
               for c in cues), f"{preset}: 不超档位上限"
    assert all(cues[i]["begin"] <= cues[i + 1]["begin"]
               for i in range(len(cues) - 1)), f"{preset}: 时间单调"
    bad = _check_token_bounds([seg], cues, "zh")
    assert not bad, f"{preset}: 切在词中间 -> {bad[:2]}"


def test_preset_distinctness():
    """档位必须有可感知差异：频繁 > 默认 > 沉浸（条数）。"""
    segments = [
        {"begin": 0, "end": 30000, "text": USER_CASE, "words": []},
        {"begin": 30000, "end": 60000,
         "text": "我们来看看这个产品在日常使用中的表现，以及它在不同场景下的续航和散热。",
         "words": []},
    ]
    counts = {p: len(export_split.split_by_preset(segments, p)) for p in BEAT_PRESETS}
    assert counts["frequent"] > counts["default"] > counts["immersive"], counts


def test_presets_avoid_bad_boundaries():
    """不把“的”“两颗|芯粒”“10|颗”等短语拆开。"""
    text = "而且它的价格并不贵但它的配置很一般，其实是由两颗芯粒组成的，包括10颗 A725 核心。"
    seg = {"begin": 0, "end": 12000, "text": text, "words": []}
    for preset in BEAT_PRESETS:
        cues = export_split.split_by_preset([seg], preset)
        for prev, nxt in zip(cues, cues[1:]):
            stop = "，。！？；：、,.!?;:"
            left = prev["text"].rstrip(stop)
            if not prev["text"].rstrip().endswith(tuple(stop)):
                assert not left.endswith(("的", "地", "得")), \
                    f"{preset}: 切在结构助词后 {prev['text']}"
                assert not (left[-1:].isdigit()
                            and nxt["text"][:1] in "颗个位名种项块张台款条只件套层期届步"), \
                    f"{preset}: 拆开数量词 {prev['text']} / {nxt['text']}"
                assert not left.endswith("两颗"), f"{preset}: 拆开量词与中心语"


def test_clause_protection_keeps_short_clause_whole():
    """默认/沉浸不切开放得下的意群；频繁可以为节奏切。"""
    text = "这个方法非常适用，它在酒店里面一般就是我们的炒菜方法。"
    seg = {"begin": 0, "end": 6000, "text": text, "words": []}
    for preset in ("default", "immersive"):
        cues = export_split.split_by_preset([seg], preset)
        for prev in cues[:-1]:
            assert prev["text"].rstrip().endswith(("，", "。", "！", "？", "；", "：")), \
                f"{preset}: 切进了完整意群 {[c['text'] for c in cues]}"


def test_quote_edges_not_split():
    """引号不能被劈在两条字幕的边界上。"""
    text = "王健林之前就说过：“我什么企业都能丢，这个不能丢。”结果他真的卖了。"
    seg = {"begin": 0, "end": 12000, "text": text, "words": []}
    for preset in BEAT_PRESETS:
        cues = export_split.split_by_preset([seg], preset)
        for prev, nxt in zip(cues, cues[1:]):
            assert not prev["text"].rstrip().endswith(("“", "‘", "（", "《")), \
                f"{preset}: 上一行以开符号结尾 {prev['text']}"
            assert not nxt["text"].lstrip().startswith(("”", "’", "）", "》")), \
                f"{preset}: 下一行以闭符号开头 {nxt['text']}"
        assert _nows("".join(c["text"] for c in cues)) == _nows(text)


def test_cross_segment_quote_and_word_boundary():
    """ASR 句界上的闭引号要回贴；单字 token 不能把“年会”拆开。

    对应真实素材里的结构：上一句以 `！` 收尾，下一句以 `”年会上...` 开头。
    按标点不能断成 `！` / `”年会上`；节奏档也不能断成 `年` / `会上`。
    """
    segments = [
        {"begin": 0, "end": 3400,
         "text": "就说出了那句著名的小目标：“比方说，", "words": []},
        {"begin": 3600, "end": 4900, "text": "我先挣他1个亿！", "words": []},
        {"begin": 6500, "end": 9000,
         "text": "”年会上还高歌了一曲《一无所有》。", "words": []},
    ]
    stream = tokenize.build_token_stream(segments, "zh")
    assert not any(t.text.lstrip().startswith("”") for t in stream[1:]), \
        "闭引号应当贴回前一个 token"
    assert _nows("".join(t.text for t in stream)) == _nows(
        "".join(s["text"] for s in segments)), "跨句 token 化文本无损"

    for preset in ("punct",) + BEAT_PRESETS:
        cues = export_split.split_by_preset(segments, preset)
        for prev, nxt in zip(cues, cues[1:]):
            assert not (prev["text"].rstrip().endswith("年")
                        and nxt["text"].lstrip().startswith("会上")), \
                f"{preset}: 把“年会”拆开了 {prev['text']} / {nxt['text']}"
        if preset == "punct":
            assert any(c["text"].rstrip().endswith("！”") for c in cues), \
                f"按标点应包含闭合引号: {[c['text'] for c in cues]}"


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        export_split.split_by_preset([{"begin": 0, "end": 1000, "text": "测试。", "words": []}],
                                     "no-such-preset")


# ---------------- 真实样本上的结构不变量 ----------------

@pytest.mark.parametrize("sample_path,preset", _real_params())
def test_invariants_on_real_samples(sample_path, preset):
    """真实识别结果上的结构不变量：无损 / 不超限 / 时间单调 / 不切词。"""
    segments = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    spec = split_presets.PRESETS[preset]
    cues = export_split.split_by_preset(segments, preset)
    tag = f"{Path(sample_path).name[:24]} {preset}"

    assert _nows("".join(c["text"] for c in cues)) == \
        _nows("".join(s["text"] for s in segments)), f"{tag}: 文本无损（{len(cues)} 条）"
    assert all(split_presets.text_width(c["text"]) / 2 <= spec.max_chars + 0.5
               for c in cues), f"{tag}: 无超限行"
    assert all(cues[i]["begin"] <= cues[i + 1]["begin"]
               for i in range(len(cues) - 1)), f"{tag}: 时间单调"
    assert all(cues[i]["end"] <= cues[i + 1]["begin"]
               for i in range(len(cues) - 1)), f"{tag}: 时间不重叠"
    bad = _check_token_bounds(segments, cues, "zh")
    assert not bad, f"{tag}: 断点都在 token 边界 {bad[:2]}"


# ---------------- API ----------------

def test_api_models(client):
    """模型与四档断句配置；不暴露字数和语言参数。"""
    r = client.get("/api/models")
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["models"]) == 1, "只保留 Fun-ASR"
    assert {m["code"] for m in data["models"]} == {"fun-asr"}
    assert data["default_model"] == "fun-asr", f"默认模型: {data['default_model']}"
    assert "languages" not in data, "英语/语言选择入口已移除"
    assert "max_chars" not in data and "split_modes" not in data, "不再提供自定义字数"
    assert data["default_preset"] == "default"
    assert {p["code"] for p in data["presets"]} == {
        "punct", "default", "frequent", "immersive"}, "返回四个断句档位"
    assert all("desc" in p for p in data["presets"]), "档位带场景说明"


def test_fun_asr_payload():
    """Fun-ASR 使用 file_urls 数组，并固定按普通话识别。"""
    payload = dashscope_client.build_transcription_payload(
        "fun-asr", "oss://audio/test.m4a")
    assert payload["model"] == "fun-asr"
    assert payload["input"] == {"file_urls": ["oss://audio/test.m4a"]}
    assert payload["parameters"] == {"language_hints": ["zh"]}
    assert "enable_words" not in payload["parameters"], "词级时间戳固定开启"


def test_api_homepage(client):
    """首页可访问。"""
    r = client.get("/")
    assert r.status_code == 200 and "音频转字幕" in r.text, "GET / 返回页面"
    assert "音频语言" not in r.text and "英语" not in r.text, "页面不再暴露语言选择"


def test_api_rejects_bad_input(client):
    """非法输入校验。"""
    r = client.post("/api/jobs", files={"file": ("a.txt", io.BytesIO(b"x"))})
    assert r.status_code == 400, "拒绝不支持的扩展名"


def test_api_export_forbidden_while_pending(client):
    """识别未完成时禁止导出。"""
    qj = jobs.create_job(file_path=config.UPLOAD_DIR / "x.mp3", filename="x.mp3",
                         model="fun-asr", language="zh")   # 不启动流水线 -> 保持 queued
    r = client.post(f"/api/jobs/{qj.id}/exports", data={"preset": "default"})
    assert r.status_code == 409, f"识别未完成时导出应返回 409，实际 {r.status_code}"


def test_api_exports_stats(client, exports):
    """四个档位都能完成导出，且只返回排版统计。"""
    for preset, meta in exports["meta"].items():
        item = exports["done"][meta["id"]]
        assert item["status"] == "done", (preset, item)
        assert item["cue_count"] > 0
        assert isinstance(item.get("stats"), dict), item
        assert "llm_stats" not in item and "max_chars" not in item


def test_export_filename_has_single_preset(client, exports):
    """导出文件名只带一次档位，不能出现“_频繁_频繁.srt”。"""
    labels = {"punct": "按标点", "default": "默认", "frequent": "频繁", "immersive": "沉浸"}
    names = [a["name"] for a in client.get("/api/assets?kind=subtitle").json()["assets"]]
    for preset, label in labels.items():
        assert any(n.endswith(f"_{label}.srt") for n in names), (preset, names)
        assert not any(f"_{label}_{label}.srt" in n for n in names), (preset, names)


def test_api_rejects_unknown_preset(client, done_job):
    """拒绝未知断句档位。"""
    r = client.post(f"/api/jobs/{done_job['id']}/exports",
                    data={"preset": "bad-mode"})
    assert r.status_code == 400, "拒绝未知断句档位"


def test_api_download_srt(client, exports):
    """SRT 下载：文本与识别结果一致，且断点都在词边界。"""
    jid = exports["meta"]["default"]["job_id"]
    eid = exports["meta"]["default"]["id"]
    r = client.get(f"/api/jobs/{jid}/exports/{eid}/download?format=srt")
    assert r.status_code == 200 and "-->" in r.text, "SRT 下载成功"
    lines = [l.strip() for l in r.text.split("\n")
             if l.strip() and "-->" not in l and not l.strip().isdigit()]
    # 不再假设"某个 segment 会整条出现"—— 意群边界是硬的，短句会被标点切开
    # （老写法假设 FAKE_SEGMENTS[0] 整条输出，架构一变就误报）。
    seg_texts = [s["text"] for s in FAKE_SEGMENTS]
    whole = "".join(seg_texts)
    assert "".join(lines) == whole, "导出文本与识别文本一致"

    # 每个断点（行累加位置）都必须落在词边界上；segment 交界本身也是合法断点
    bad = _check_token_bounds(
        FAKE_SEGMENTS,
        [{"text": line, "begin": 0, "end": 1000} for line in lines], "zh")
    assert not bad, f"断点切在词中间: {bad[:2]}"


def test_api_download_punct(client, exports):
    """标点直断：无标点句整句一条，有标点句按子句拆分。"""
    jid = exports["meta"]["punct"]["job_id"]
    eid = exports["meta"]["punct"]["id"]
    r = client.get(f"/api/jobs/{jid}/exports/{eid}/download?format=srt")
    lines = [l.strip() for l in r.text.split("\n")
             if l.strip() and "-->" not in l and not l.strip().isdigit()]
    assert FAKE_SEGMENTS[1]["text"] in lines, "标点直断：无标点句整句一条"
    assert "你好，" in lines and "世界。" in lines, "标点直断：按标点拆分子句"


def test_api_download_txt_and_404(client, exports):
    """TXT 下载、沉浸档下载、不存在的导出返回 404。"""
    jid = exports["meta"]["default"]["job_id"]
    r = client.get(f"/api/jobs/{jid}/exports/{exports['meta']['default']['id']}/download?format=txt")
    assert r.status_code == 200 and "笔记本电脑" in r.text, "TXT 下载成功"
    r = client.get(f"/api/jobs/{jid}/exports/{exports['meta']['immersive']['id']}/download?format=srt")
    assert r.status_code == 200, "沉浸档下载成功"
    r = client.get(f"/api/jobs/{jid}/exports/nope/download")
    assert r.status_code == 404, "不存在的导出返回 404"


def test_api_asset_file_range(client, done_job):
    """资产文件必须支持 Range。

    播放器只有在响应带 206 / Accept-Ranges 时才会拿到 seekable 区间，
    否则给它设 currentTime 会被静默忽略 —— 表现为点字幕不跳转、播放不高亮。
    顺带守住 download 参数：此前它被忽略，连播放请求都带 attachment。
    """
    aud = [a for a in client.get("/api/assets").json()["assets"]
           if a["kind"] == "audio"]
    assert aud, "已完成任务应登记一条音频资产"
    url = f"/api/assets/{aud[0]['id']}/file"

    r = client.get(url)
    assert r.status_code == 200 and r.content == b"\x00" * 4096, "完整读取"
    assert r.headers["accept-ranges"] == "bytes", "声明支持 Range"
    assert r.headers["content-disposition"].startswith("inline"), "播放走 inline"

    r = client.get(url, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206, "分段请求返回 206"
    assert r.headers["content-range"] == "bytes 0-99/4096", r.headers.get("content-range")
    assert len(r.content) == 100

    r = client.get(url, headers={"Range": "bytes=4000-"})
    assert r.status_code == 206 and len(r.content) == 96, "开放式 Range 取到末尾"

    r = client.get(url, headers={"Range": "bytes=-100"})
    assert r.status_code == 206 and len(r.content) == 100, "末尾 N 字节"

    r = client.get(url, headers={"Range": "bytes=99999-"})
    assert r.status_code == 416, "越界返回 416"

    r = client.get(url + "?download=1")
    assert r.headers["content-disposition"].startswith("attachment"), "下载走 attachment"


def test_api_job_list(client, exports):
    """任务列表可见任务与导出数。"""
    jid = exports["meta"]["default"]["job_id"]
    r = client.get("/api/jobs")
    assert any(j["id"] == jid and j["export_count"] == 4
               for j in r.json()["jobs"]), "列表中可见任务与导出数"


def test_api_import(client):
    """导入识别结果：直接完成、按普通话处理、可立即导出；非法输入被拒。"""
    payload = json.dumps(FAKE_SEGMENTS).encode("utf-8")
    r = client.post("/api/jobs/import",
                    files={"file": ("测试音频1.json", io.BytesIO(payload))})
    assert r.status_code == 200, r.text
    ij = r.json()
    assert ij["status"] == "done" and ij["model_name"] == "导入的识别结果", "导入任务直接完成"
    assert ij["detected_lang"] == "zh", "固定按普通话处理"
    assert len(ij["preview"]) == 2, "携带识别预览"
    r = client.post(f"/api/jobs/{ij['id']}/exports", data={"preset": "punct"})
    assert r.status_code == 200, "导入任务可直接导出"

    assert client.post("/api/jobs/import",
                       files={"file": ("a.txt", io.BytesIO(b"x"))}).status_code == 400, \
        "拒绝非 json 文件"
    assert client.post("/api/jobs/import",
                       files={"file": ("a.json", io.BytesIO(b"{}"))}).status_code == 400, \
        "拒绝非法结构"
    assert client.post(
        "/api/jobs/import",
        files={"file": ("a.json", io.BytesIO(b'[{"text": "x"}]'))}).status_code == 400, \
        "拒绝缺 begin/end 的分段"


# ---------------- 本机访问控制 ----------------

def test_host_header_parsing():
    """Host 解析：端口、大小写、IPv6、尾点都算本机；子域与外部域名都不算。"""
    for ok in ("127.0.0.1", "127.0.0.1:18080", "localhost", "LOCALHOST:18080",
               "[::1]:18080", "127.0.0.1."):
        assert main._is_local_host(ok), f"{ok} 应判为本机"
    for bad in ("evil.com", "127.0.0.1.evil.com", "evil.com:18080", "", "0.0.0.0"):
        assert not main._is_local_host(bad), f"{bad} 不应判为本机"


def test_api_rejects_non_local_host():
    """Host 不是本机就一律 403——这是挡 DNS rebinding 的那道，读接口同样要挡。"""
    evil = TestClient(main.app, base_url="http://evil.com")
    assert evil.get("/api/models").status_code == 403
    assert evil.get("/api/assets").status_code == 403
    r = evil.post("/api/jobs", headers={"origin": "http://evil.com"},
                  files={"file": ("a.mp3", io.BytesIO(b"\x00" * 128))})
    assert r.status_code == 403, f"跨站上传未被拦截: {r.status_code}"


def test_api_accepts_local_hosts():
    """127.0.0.1 与 localhost 都放行。"""
    for base in ("http://127.0.0.1:18080", "http://localhost:18080"):
        assert TestClient(main.app, base_url=base).get("/api/models").status_code == 200


# ---------------- 资源回收 ----------------

def test_cleanup_stale_uploads(tmp_path, monkeypatch):
    """启动时清掉残留的上传临时文件，正常文件不动。"""
    up = tmp_path / "up"
    up.mkdir()
    (up / ".tmp_deadbeef_a.mp3").write_bytes(b"x")
    (up / ".tmp_deadbeef_b.mp3").write_bytes(b"y")
    (up / "keep.mp3").write_bytes(b"z")
    monkeypatch.setattr(config, "UPLOAD_DIR", up)
    assert main._cleanup_stale_uploads() == 2
    assert [p.name for p in up.iterdir()] == ["keep.mp3"]


def test_export_leaves_no_temp_dirs(client, done_job):
    """导出不再往 results/ 写东西（以前每导一次就留一个空目录）。"""
    def snap():
        return ({p.name for p in config.RESULT_DIR.rglob("*")}
                if config.RESULT_DIR.exists() else set())

    before = snap()
    eid = client.post(f"/api/jobs/{done_job['id']}/exports",
                      data={"preset": "default"}).json()["id"]
    for _ in range(50):
        e = next(x for x in client.get(f"/api/jobs/{done_job['id']}/exports").json()["exports"]
                 if x["id"] == eid)
        if e["status"] != "pending":
            break
        time.sleep(0.1)
    assert e["status"] == "done", e
    assert snap() == before, f"results/ 被写入了: {snap() - before}"


def test_jobs_pruned():
    """内存里的已完结任务会剪枝，不会随反复导出无限堆积。"""
    assert jobs._MAX_JOBS > 0
    for i in range(jobs._MAX_JOBS + 12):
        j = jobs.create_job(Path("x.mp3"), f"{i}.mp3", "fun-asr", "zh")
        j.status = "done"
    assert len(jobs._JOBS) <= jobs._MAX_JOBS, f"未剪枝: {len(jobs._JOBS)}"


def test_jobs_prune_keeps_running_export():
    """还有导出在跑的任务不会被剪掉（最老也不剪）。"""
    keep = jobs.create_job(Path("x.mp3"), "keep.mp3", "fun-asr", "zh")
    keep.status = "done"
    keep.exports.append(jobs.Export(id="e1", job_id=keep.id, preset="default"))
    for i in range(jobs._MAX_JOBS + 5):
        j = jobs.create_job(Path("x.mp3"), f"{i}.mp3", "fun-asr", "zh")
        j.status = "done"
    assert jobs.get_job(keep.id) is not None, "有进行中导出的任务被误剪"


# ---------------- 资产库 ----------------

def test_legacy_asset_origin_hides_old_mode():
    """历史 rule/llm 字幕只显示“旧版断句”，不再暴露两套旧实现。"""
    asset = assets_mod.Asset(
        id="x" * 12, kind=assets_mod.KIND_SUBTITLE, name="a.srt",
        file="subtitle/x_a.srt", model="fun-asr", split_mode="rule")
    assert "旧版断句" in asset.origin_text()
    assert "rule" not in asset.origin_text()


def test_asset_store_rebuilds_index_from_disk(tmp_path):
    """index.json 丢了也能靠目录扫描把资产捞回来（索引只是缓存）。"""
    root = tmp_path / "assets"
    store = assets_mod.AssetStore(root)
    a = store.add_segments([{"begin": 0, "end": 900, "text": "测试。", "words": []}],
                           "样本", "zh", "fun-asr")
    store.index_path.unlink()

    reborn = assets_mod.AssetStore(root)
    assert reborn.load() == 1, "应从目录扫描恢复出 1 条资产"
    back = reborn.get(a.id)
    assert back is not None and back.file == a.file, back
    assert back.cue_count == 1, "条数应从文件量出来"
    assert reborn.index_path.exists(), "重建后应把索引写回磁盘"


def test_asset_delete_default_keeps_descendants(client, tmp_path):
    """默认不级联：只删自己，派生资产留下（音画联动会失效但不静默删数据）。"""
    src = tmp_path / "a.mp3"
    src.write_bytes(b"\x00" * 32)
    audio = jobs.STORE.add_audio(src, "a.mp3", "zh", "fun-asr")
    seg = jobs.STORE.add_segments([{"begin": 0, "end": 900, "text": "测试。", "words": []}],
                                  "a", "zh", "fun-asr", source_id=audio.id)

    listed = {a["id"]: a for a in client.get("/api/assets").json()["assets"]}
    assert listed[audio.id]["derived_count"] == 1, listed[audio.id]
    assert listed[seg.id]["derived_count"] == 0, listed[seg.id]

    assert client.delete(f"/api/assets/{audio.id}").json()["removed"] == [audio.id]
    assert jobs.STORE.get(audio.id) is None
    assert jobs.STORE.get(seg.id) is not None, "默认不应连带删除派生资产"

    assert client.delete(f"/api/assets/{seg.id}?cascade=1").json()["removed"] == [seg.id]
    assert jobs.STORE.get(seg.id) is None
    assert client.delete("/api/assets/nope").status_code == 404


def test_asset_delete_cascade(client, tmp_path):
    """cascade=1：整条链（音频 -> 识别结果 -> 字幕）一起删干净。"""
    src = tmp_path / "b.mp3"
    src.write_bytes(b"\x00" * 32)
    audio = jobs.STORE.add_audio(src, "b.mp3", "zh", "fun-asr")
    seg = jobs.STORE.add_segments([{"begin": 0, "end": 900, "text": "测试。", "words": []}],
                                  "b", "zh", "fun-asr", source_id=audio.id)

    removed = client.delete(f"/api/assets/{audio.id}?cascade=1").json()["removed"]
    assert sorted(removed) == sorted([audio.id, seg.id]), removed
    assert jobs.STORE.get(audio.id) is None and jobs.STORE.get(seg.id) is None


def test_api_import_always_zh(client):
    """导入结果一律按普通话落库，不再按文本内容判定外语。"""
    segs = [{"begin": 0, "end": 2000, "text": "Hello world, this is a test."}]
    r = client.post("/api/jobs/import",
                    files={"file": ("legacy.json", io.BytesIO(json.dumps(segs).encode("utf-8")))})
    assert r.status_code == 200, r.text
    assert r.json()["detected_lang"] == "zh", r.json()
    listed = client.get("/api/assets?kind=segments").json()["assets"]
    assert listed and listed[0]["language"] == "zh", listed


# ---------------- 断句收尾 ----------------

def test_finalize_no_overlap():
    """最短时长不得把 end 顶过下一条的 begin（处理顺序敏感）。"""
    out = split_base.finalize([
        {"text": "a", "begin": 1000, "end": 1010},
        {"text": "b", "begin": 1015, "end": 3000},
    ])
    assert out[0]["end"] <= out[1]["begin"], out
    assert out[0]["end"] > out[0]["begin"], out


def test_finalize_normal_cues_untouched():
    """正常间隔的字幕不被改动。"""
    src = [{"text": "a", "begin": 0, "end": 1000},
           {"text": "b", "begin": 1000, "end": 2500}]
    assert split_base.finalize([dict(c) for c in src]) == src


if __name__ == "__main__":
    sys.exit(0 if pytest.main([__file__, "-v", "-p", "no:cacheprovider"]) == 0 else 1)
