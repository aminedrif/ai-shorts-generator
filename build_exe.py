import os
import sys
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def build():
    print("=" * 60)
    print("Building AI Shorts Generator executable...")
    print("=" * 60)

    # Clean old build/dist directories and spec file
    for d in ["build", "dist"]:
        p = BASE_DIR / d
        if p.exists():
            print(f"Cleaning {p}...")
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception as e:
                print(f"Warning: could not clean {p}: {e}")

    spec_file = BASE_DIR / "AIShortsGenerator.spec"
    if spec_file.exists():
        try:
            spec_file.unlink()
        except Exception:
            pass

    # PyInstaller command
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--name", "AIShortsGenerator",
        "--add-data", f"{BASE_DIR / 'web'}{os.pathsep}web",
        "--add-data", f"{BASE_DIR / 'src' / 'cascades'}{os.pathsep}src/cascades",
        "--collect-all", "faster_whisper",
        "--collect-all", "ctranslate2",
        "--collect-all", "uvicorn",
        "--collect-all", "yt_dlp",
        "--collect-all", "google.generativeai",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "dotenv",
        "--hidden-import", "cv2",
        "--hidden-import", "imageio_ffmpeg",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        "--exclude-module", "PySide2",
        "--exclude-module", "PySide6",
        "--exclude-module", "matplotlib",
        "--exclude-module", "IPython",
        "--exclude-module", "notebook",
        "--exclude-module", "tkinter",
        str(BASE_DIR / "server.py"),
    ]

    print("Running PyInstaller...")
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    if res.returncode != 0:
        print(f"PyInstaller build failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    dist_dir = BASE_DIR / "dist" / "AIShortsGenerator"
    print(f"\nPost-processing distribution directory: {dist_dir}")

    # 1. Copy ffmpeg.exe from imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if ffmpeg_src.exists():
            target_ffmpeg = dist_dir / "ffmpeg.exe"
            print(f"Copying FFmpeg binary ({ffmpeg_src.name}) -> {target_ffmpeg}...")
            shutil.copy2(ffmpeg_src, target_ffmpeg)
    except Exception as e:
        print(f"Warning: Could not bundle ffmpeg.exe automatically: {e}")

    # 2. Copy web folder alongside exe for easy editing
    web_target = dist_dir / "web"
    if not web_target.exists():
        print(f"Copying web folder -> {web_target}...")
        shutil.copytree(BASE_DIR / "web", web_target, dirs_exist_ok=True)

    # Copy bundled cascades
    cascades_target = dist_dir / "src" / "cascades"
    cascades_target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE_DIR / "src" / "cascades", cascades_target, dirs_exist_ok=True)

    # 3. Copy .env or .env.example
    env_target = dist_dir / ".env"
    if (BASE_DIR / ".env").exists() and not env_target.exists():
        print(f"Copying .env -> {env_target}...")
        shutil.copy2(BASE_DIR / ".env", env_target)
    elif (BASE_DIR / ".env.example").exists() and not env_target.exists():
        print(f"Copying .env.example -> {env_target}...")
        shutil.copy2(BASE_DIR / ".env.example", env_target)

    # Always ensure .env.example is there too as reference
    if (BASE_DIR / ".env.example").exists():
        shutil.copy2(BASE_DIR / ".env.example", dist_dir / ".env.example")

    # 4. Create Run_AI_Shorts_Generator.bat
    bat_content = """@echo off
title AI Shorts Generator
cd /d "%~dp0"
echo ========================================================
echo   Starting AI Shorts Generator...
echo   Open your browser at: http://localhost:8000
echo ========================================================
AIShortsGenerator.exe
pause
"""
    (dist_dir / "Run_AI_Shorts_Generator.bat").write_text(bat_content, encoding="utf-8")

    # 5. Create README.txt
    readme_content = """AI SHORTS GENERATOR - STANDALONE PORTABLE VERSION
==================================================

HOW TO RUN:
1. Double-click "Run_AI_Shorts_Generator.bat" or "AIShortsGenerator.exe".
2. A terminal window will open showing the server logs.
3. Your default web browser will automatically open:
   http://localhost:8000
4. Keep the terminal window open while using the application.
   To stop the application, simply close the terminal window or press Ctrl+C.

API KEY CONFIGURATION (.env):
- Open the ".env" file with Notepad.
- Enter your Gemini API key (free at https://aistudio.google.com):
  GEMINI_API_KEY=your_api_key_here
- Or OpenAI API key if using OpenAI:
  OPENAI_API_KEY=your_api_key_here

SHARING WITH YOUR FRIEND:
- You can zip the entire "AIShortsGenerator" folder (or use 7-Zip / WinRAR).
- Send the .zip file to your friend.
- All your friend has to do is extract the zip, edit .env to put their Gemini API key, and double-click "Run_AI_Shorts_Generator.bat"!
- FFmpeg and all dependencies are already bundled inside. No Python or ffmpeg install needed!

OUTPUT VIDEOS:
- All generated short videos are automatically saved inside the "output" folder.
"""
    (dist_dir / "README.txt").write_text(readme_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print("BUILD SUCCESSFUL!")
    print(f"The portable folder is ready at: {dist_dir}")
    print("=" * 60)

if __name__ == "__main__":
    build()
