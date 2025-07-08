@echo off
setlocal enabledelayedexpansion

REM Get the script's directory
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%\venv"

REM Remove trailing backslash if present
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Check if venv exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment at %VENV_DIR%
    python -m venv "%VENV_DIR%"
)

REM Activate the venv
call "%VENV_DIR%\Scripts\activate.bat"

REM Run the Python script
python "%SCRIPT_DIR%\app.py" %*

endlocal
