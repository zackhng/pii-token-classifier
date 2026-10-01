@echo off
rem Full training run, detached-friendly. Log: outputs\train.log
cd /d "%~dp0src"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
set TRANSFORMERS_VERBOSITY=error
"..\.venv\Scripts\python.exe" -u train.py %* > "..\outputs\train.log" 2>&1
