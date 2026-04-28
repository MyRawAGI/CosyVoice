@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PATH=%PATH%;C:\Users\MyRawAGI\AppData\Local\Microsoft\WinGet\Links

call C:\Users\MyRawAGI\miniconda3\Scripts\activate.bat cosyvoice
cd /d "%~dp0"

python novel_tts.py %*
pause
