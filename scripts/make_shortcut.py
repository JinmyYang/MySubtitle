"""在桌面生成「本地中文字幕王」快捷方式（指向项目根目录的 .bat）。

为什么要单独写个脚本：创建 .lnk 需要 WScript.Shell 这个 COM 对象，
PowerShell 工具会以"COM 可执行任意代码"为由拦掉；pywin32 在本机是装好的，
用它做同样的事既干净、又能留个可复现的记录——以后挪了目录或换了机器，
跑一次 `python scripts/make_shortcut.py` 就有桌面图标了。

用法: python scripts/make_shortcut.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 本项目是从「启动中文字幕王」（云端 Fun-ASR 版）分出来的本地 ASR 版，
# 名字上刻意区分，两个桌面快捷方式互不覆盖。
BAT_NAME = "本地中文字幕王.bat"
SHORTCUT_NAME = "本地中文字幕王"


def desktop_dir() -> Path:
    """定位桌面。优先 OneDrive 重定向后的那个（存在就用），否则用用户目录下的。"""
    for key in ("OneDrive", "USERPROFILE"):
        base = os.environ.get(key)
        if base:
            d = Path(base) / "Desktop"
            if d.is_dir():
                return d
    return Path.home() / "Desktop"


def main() -> int:
    target = ROOT / BAT_NAME
    if not target.is_file():
        print(f"[错误] 找不到 {target}", flush=True)
        return 1

    import pythoncom
    from win32com.client import Dispatch

    desktop = desktop_dir()
    lnk_path = desktop / f"{SHORTCUT_NAME}.lnk"

    pythoncom.CoInitialize()
    try:
        shell = Dispatch("WScript.Shell")
        sc = shell.CreateShortcut(str(lnk_path))
        sc.TargetPath = str(target)
        sc.WorkingDirectory = str(ROOT)     # 批处理里虽有 %~dp0，这里写对更稳
        sc.Description = "本地中文字幕王（本地 ASR 版）：一键启动服务并打开页面"
        sc.WindowStyle = 1                  # 1 = 正常窗口；关窗即停服务
        sc.Save()
    finally:
        pythoncom.CoUninitialize()

    print(f"[完成] {lnk_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
