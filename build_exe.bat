@echo off
setlocal
cd /d "%~dp0"
py -3 -m pip install pyinstaller
py -3 -m PyInstaller --noconfirm --clean --windowed --onefile --name "STL Texture Applier" --collect-all pyvista --collect-all pyvistaqt --collect-all vtk --collect-submodules app --hidden-import PIL --hidden-import trimesh Finish.pyw
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
echo Built dist\STL Texture Applier.exe
endlocal
