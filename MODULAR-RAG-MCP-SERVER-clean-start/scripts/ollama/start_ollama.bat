@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"

cd /d "%PROJECT_ROOT%" || exit /b 1

if not exist ".venv\Scripts\activate.bat" (
    echo Error: virtual environment .venv was not found. 1>&2
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python "scripts\ollama\start_ollama.py" %*

exit /b %ERRORLEVEL%
