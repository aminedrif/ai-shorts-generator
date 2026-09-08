import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    # Project paths
    base_dir: Path = BASE_DIR
    output_dir: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
    temp_dir: Path = BASE_DIR / os.getenv("TEMP_DIR", "temp")

    logs_dir: Path = BASE_DIR / os.getenv("LOGS_DIR", "logs")

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
    If FFmpeg is not found in system PATH, falls back to imageio_ffmpeg,
    creates a standardized ffmpeg.exe binary in the temp bin directory,
    and prepends it to PATH so external subprocesses and yt-dlp discover it.
    """
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
            target_exe = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
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


# Initialize ffmpeg environment at import time
get_ffmpeg_bin()

