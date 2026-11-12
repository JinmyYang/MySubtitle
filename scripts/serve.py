"""启动本地服务：自动挑一个能绑定的回环端口，服务就绪后再打开浏览器。

**首选 18080，不要改回 8000**：8000 与 3000 是最容易被本机其他应用强占的两个端口
（实测踩过，两者无法同时启动）。18080 在 TCP 动态范围（1024–15000）与 Windows 默认范围（49152–65535）之外，
不会被 WinNAT / Hyper-V 的保留区顺手占掉。

为什么不能写死端口：Windows（Hyper-V / WSL / Docker）会动态保留一段 TCP 端口，
用 `netsh interface ipv4 show excludedportrange protocol=tcp` 可以查到。端口一旦落进
保留区，bind 会直接抛 WinError 10013，浏览器只能停在"无法访问"，而 `netstat` 里
看不到任何进程占用——很容易误判成"端口空着，服务没起来"。

所以启动流程是：
    1. 先看有没有已经在跑的本服务（有就只打开页面，不起第二个实例）；
    2. 依次尝试候选端口，逐个真绑定一次，绑不上就跳过；
    3. 候选都不行就让系统分配一个（port 0），保证一定能起来。
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402 - 必须先把项目根目录塞进 sys.path

HOST = "127.0.0.1"
# 18080 是首选：留给本项目专用，避开最常被占用的 8000 / 3000。
# 其余候选都在高段，且刻意不碰 8000/3000。端口保留区是会变的，所以靠"真绑一次"判断。
CANDIDATE_PORTS = (18080, 18081, 18088, 18090, 28080)
STATE_FILE = ROOT / ".runtime" / "server.json"


def _probe(port: int, timeout: float = 0.4) -> bool:
    """该端口上是否已经跑着本服务。

    只判断"端口通"不够稳妥——别的程序占着也会通；这里比对 /api/models 的响应内容，
    确认是自己的实例才复用，避免把用户领到别人的页面上。
    """
    try:
        conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
        try:
            conn.request("GET", "/api/models")
            resp = conn.getresponse()
            body = resp.read(4096)
        finally:
            conn.close()
        return resp.status == 200 and b'"presets"' in body
    except (OSError, http.client.HTTPException, ValueError):
        return False


def _read_state_port() -> int | None:
    """读上次启动记录的端口（可能是用户手动起的、也可能上次没退干净）。"""
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        port = int(data.get("port"))
        return port if 0 < port < 65536 else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _write_state(port: int) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps({"port": port, "pid": os.getpid()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass          # 记录失败只影响"再次双击只开页面"这个便利，不影响启动


def _clear_state(port: int) -> None:
    """只删自己写的那条，避免把新实例的记录误删。"""
    if _read_state_port() != port:
        return
    try:
        STATE_FILE.unlink()
    except OSError:
        pass


def _find_running() -> int | None:
    """找一个已在运行的本服务实例。"""
    seen: set[int] = set()
    for port in filter(None, (_read_state_port(), *CANDIDATE_PORTS)):
        if port in seen:
            continue
        seen.add(port)
        if _probe(port):
            return port
    return None


def _bind(port: int) -> socket.socket | None:
    """真绑一次。绑得上才算这个端口可用。

    刻意不设 SO_REUSEADDR：在 Windows 上它允许抢占别人已绑定的端口，
    结果是"看着启动成功，请求却进了另一个程序"。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, port))
        sock.listen(128)
        return sock
    except OSError as exc:
        sock.close()
        print(f"  [跳过] 端口 {port} 绑不上：{exc.strerror or exc}", flush=True)
        return None


def _pick_socket() -> tuple[socket.socket, int]:
    """按候选顺序挑端口；全都不行就交给系统分配。"""
    want = os.environ.get("PORT", "").strip()
    ports = ([int(want)] if want.isdigit() else []) + [
        p for p in CANDIDATE_PORTS if p != int(want or 0)
    ]
    for port in ports:
        sock = _bind(port)
        if sock is not None:
            return sock, port

    print("  [提示] 候选端口全部不可用，改由系统分配", flush=True)
    sock = _bind(0)
    if sock is None:      # port 0 理论上不会失败，兜底还是给个明确报错
        raise SystemExit("[错误] 无法创建监听套接字")
    return sock, sock.getsockname()[1]


def _open_when_ready(port: int, timeout: float = 30.0) -> None:
    """等服务真的能应答再打开浏览器，避免抢在启动完成前访问拿到空白页。"""
    url = f"http://{HOST}:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe(port, timeout=0.5):
            print(f"[就绪] {url}", flush=True)
            if os.environ.get("NO_BROWSER"):    # 调试/手工起服务时不想弹窗口
                return
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - 打不开浏览器不影响服务本身
                pass
            return
        time.sleep(0.25)
    print(f"[提示] 服务已启动，但浏览器未自动打开，请手动访问 {url}", flush=True)


def main() -> int:
    running = _find_running()
    if running:
        url = f"http://{HOST}:{running}/"
        print(f"[提示] 服务已在运行（端口 {running}），直接打开页面。", flush=True)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            print(f"        请手动访问 {url}", flush=True)
        return 0

    sock, port = _pick_socket()
    from app.main import app  # 延迟导入：导入失败要能落到下面的报错里

    url = f"http://{HOST}:{port}/"
    print("=" * 44)
    print("  音频转字幕工具（阿里云百炼 ASR）")
    print(f"  地址：{url}")
    print("  关闭本窗口即停止服务")
    print("=" * 44, flush=True)

    config = uvicorn.Config(app, host=HOST, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    _write_state(port)
    threading.Thread(target=_open_when_ready, args=(port,), daemon=True).start()
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass
    finally:
        _clear_state(port)
        try:
            sock.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
