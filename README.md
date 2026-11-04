# STL Texture Applier

Desktop app that **carves a surface texture into an STL**. The bottom face stays smooth so the part still prints flat on the bed.

## Download

Windows one-file build: [STL-Texture-Applier.exe on the latest release](https://github.com/epjarv/STL-Texture-Applier/releases/latest).

## Run from source

```powershell
py -3 -m pip install -r requirements.txt
.\Finish.bat
```

Or `py -3 -m app` after dependencies are installed.

## Build exe

```powershell
.\build_exe.bat
```

Output: `dist\STL Texture Applier.exe` (release asset name may be `STL-Texture-Applier.exe`).

## Use it

1. Open an STL.
2. Choose a texture pattern and depth.
3. Preview, then export a new STL (`Output STL`).

## Version

Matches repo tag on `main` (currently `v1.0.1`).
