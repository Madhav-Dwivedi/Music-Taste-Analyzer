@echo off
cd /d "%~dp0"
title Music Taste Analyzer - Local AI Web Studio
color 0b
mode con cols=120 lines=40

where python >nul 2>nul
if errorlevel 1 (
    echo [Error] Python was not found in PATH. Please install Python 3.10+ and check "Add Python to PATH".
    pause
    exit /b 1
)

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [System] Initializing centralized AI Environment. This will take a few minutes...
    python -m venv .venv
    if errorlevel 1 (
        echo [Error] Failed to create virtual environment .venv.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
)

".venv\Scripts\python.exe" -c "import hashlib, pathlib, sys; cur=hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(); st=pathlib.Path('.venv/.requirements.sha256'); sys.exit(0 if st.exists() and st.read_text(encoding='utf-8').strip()==cur else 1)"
if errorlevel 1 (
    echo [System] Installing/updating dependencies from requirements.txt...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [Error] Failed to install dependencies from requirements.txt.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -c "import hashlib, pathlib; pathlib.Path('.venv/.requirements.sha256').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(), encoding='utf-8')"
)

".venv\Scripts\python.exe" web_app.py
pause
