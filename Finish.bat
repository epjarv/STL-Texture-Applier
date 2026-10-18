@echo off
setlocal
cd /d "%~dp0"

set "PY=py"
where py >nul 2>&1 || set "PY=python"

%PY% -3 -c "import PySide6, pyvistaqt, trimesh, numpy, PIL" 1>nul 2>nul
if errorlevel 1 (
  echo Installing STL Texture Applier. The first launch takes a minute.
  %PY% -3 -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo.
    echo Install failed. Install Python 3 from https://www.python.org/downloads/ and run this again.
    pause
    exit /b 1
  )
)

where pyw >nul 2>&1
if errorlevel 1 (
  %PY% -3 -m app
) else (
  start "" pyw -3 -m app
)
if errorlevel 1 pause
endlocal
