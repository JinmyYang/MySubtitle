@echo off
rem ============================================
rem  中文字幕王（音频转字幕工具）- 一键启动脚本
rem  双击本文件即可：检查环境 -> 启动服务 -> 自动打开网页
rem  端口由 scripts\serve.py 自动挑选：首选 18080（避开 Applio Desktop 的 8000/3000），
rem  绑不上会依次退到 18081 / 18088 / 18090 / 28080
rem  关闭本窗口 = 停止服务；再次双击即重新启动
rem  其它测试脚本（tests\test_api.py、scripts\smoke_test.py 等）按需手跑
rem ============================================
chcp 65001 >nul
title 本地中文字幕王
cd /d "%~dp0"

echo ============================================
echo   音频转字幕工具（阿里云百炼 ASR）
echo ============================================
echo.

rem 1. 检查 Python
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.10+ 并勾选加入 PATH。
    pause
    exit /b 1
)

rem 2. 检查百炼 API Key（交给 python 实际解析，避免 cmd 读不到用户变量时误报）
python -c "import sys; sys.path.insert(0,'.'); from app import config; sys.exit(0 if config.API_KEY else 1)" >nul 2>nul
if errorlevel 1 (
    echo [警告] 未检测到 DASHSCOPE_API_KEY，识别功能将不可用。
    echo        请在系统环境变量中配置百炼 API Key 后重新双击本脚本。
    echo        按任意键仍可启动（仅能查看页面）。
    pause >nul
)

rem 3. 首次运行自动补装依赖（jieba 也要查，缺了服务起不来）
python -c "import fastapi, uvicorn, httpx, multipart, jieba" >nul 2>nul
if errorlevel 1 (
    echo [提示] 首次运行，正在安装依赖...
    rem 用 python -m pip 而不是裸 pip：PATH 里可能同时存在多个 Python，
    rem 裸 pip 装到的解释器未必就是上面检查的那一个。
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络后重试。
        pause
        exit /b 1
    )
)

rem 4. 启动服务（端口选择、服务就绪后打开浏览器都在 serve.py 里）
python scripts\serve.py
set EXITCODE=%errorlevel%
if not "%EXITCODE%"=="0" (
    echo.
    echo [错误] 服务异常退出（代码 %EXITCODE%），错误信息见上方。
    pause
)
exit /b %EXITCODE%
