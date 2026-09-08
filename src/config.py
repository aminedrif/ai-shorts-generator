import os
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

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()


def get_ffmpeg_bin() -> str:
    """Finds ffmpeg binary from PATH or falls back to imageio_ffmpeg."""
    import shutil
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

