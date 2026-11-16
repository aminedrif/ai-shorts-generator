import os
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
    # If running inside Electron extraResources (.../resources/backend),
    # also check for .env in the top-level application directory next to A-S-G Studio.exe
    if BASE_DIR.parent.name == "resources":
        electron_app_dir = BASE_DIR.parent.parent
        if (electron_app_dir / ".env").exists():
            load_dotenv(electron_app_dir / ".env")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env next to the executable/project
load_dotenv(BASE_DIR / ".env")
load_dotenv()

def resolve_app_root() -> Path:
    if getattr(sys, "frozen", False) and BASE_DIR.parent.name == "resources":
        return BASE_DIR.parent.parent
    return BASE_DIR

APP_ROOT = resolve_app_root()


@dataclass
class AppConfig:
    # Project paths
    base_dir: Path = BASE_DIR
    app_root: Path = APP_ROOT
    output_dir: Path = APP_ROOT / os.getenv("OUTPUT_DIR", "output")
    temp_dir: Path = APP_ROOT / os.getenv("TEMP_DIR", "temp")
    logs_dir: Path = APP_ROOT / os.getenv("LOGS_DIR", "logs")

    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Whisper Settings
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # Video Settings
    default_aspect_ratio: str = os.getenv("DEFAULT_ASPECT_RATIO", "9:16")
    subtitle_font_size: int = int(os.getenv("SUBTITLE_FONT_SIZE", "24"))
    video_encoder: str = os.getenv("VIDEO_ENCODER", "auto")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()

_NVENC_AVAILABLE = None


def is_nvenc_available() -> bool:
    """Checks whether NVIDIA NVENC hardware acceleration is supported by FFmpeg."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        res = subprocess.run(
            [get_ffmpeg_bin(), "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in res.stdout
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def get_ffmpeg_bin() -> str:
    """
    Resolves the FFmpeg binary path.
    1. Checks if ffmpeg.exe exists directly in BASE_DIR or BASE_DIR/bin
    2. Checks PyInstaller bundle directory (sys._MEIPASS)
    3. Checks system PATH
    4. Falls back to imageio_ffmpeg, creating a copy in temp/bin
    Ensures FFmpeg directory is in os.environ['PATH'] so yt-dlp finds it.
    """
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = [
        config.base_dir / exe_name,
        config.base_dir / "bin" / exe_name,
    ]
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / exe_name)

    for c in candidates:
        if c.is_file():
            bin_folder_str = str(c.parent)
            if bin_folder_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_folder_str + os.pathsep + os.environ.get("PATH", "")
            return str(c)

    import shutil
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path

    try:
        import imageio_ffmpeg
        source_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if source_exe.exists():
            bin_dir = config.temp_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            target_exe = bin_dir / exe_name
            if not target_exe.exists() or target_exe.stat().st_size != source_exe.stat().st_size:
                try:
                    shutil.copy2(source_exe, target_exe)
                except Exception:
                    pass

            effective_exe = target_exe if target_exe.exists() else source_exe
            bin_folder_str = str(effective_exe.parent)
            if bin_folder_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_folder_str + os.pathsep + os.environ.get("PATH", "")
            return str(effective_exe)
    except Exception:
        pass

    return "ffmpeg"


def setup_cuda_dll_paths():
    """
    On Windows, locates NVIDIA CUDA/cuBLAS runtime DLLs from installed
    Python packages or frozen bundle and registers them with os.add_dll_directory
    and PATH so CTranslate2 and faster-whisper can find cublas64_12.dll.
    """
    if os.name != "nt":
        return

    import site

    dll_paths = set()
    search_dirs = []
    if getattr(sys, "frozen", False):
        search_dirs.append(Path(sys.executable).resolve().parent)
        if hasattr(sys, "_MEIPASS"):
            search_dirs.append(Path(sys._MEIPASS))
    try:
        search_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    search_dirs.append(os.path.dirname(sys.executable))

    for base in search_dirs:
        p = Path(base)
        nvidia_dir = p / "nvidia"
        if nvidia_dir.is_dir():
            for sub in nvidia_dir.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir():
                    dll_paths.add(str(bin_dir))
        internal_dir = p / "_internal"
        if internal_dir.is_dir():
            dll_paths.add(str(internal_dir))

    for path_str in dll_paths:
        try:
            os.add_dll_directory(path_str)
        except Exception:
            pass
        if path_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")


# Initialize environment at import time
get_ffmpeg_bin()
setup_cuda_dll_paths()

