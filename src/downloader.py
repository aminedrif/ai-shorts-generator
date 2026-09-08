import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import yt_dlp
from src.config import config, get_ffmpeg_bin


def sanitize_filename(name: str) -> str:
    """Removes invalid filename characters and trims length."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r"\s+", "_", clean).strip("._")
    return clean[:80] if clean else "video"


class VideoDownloader:
    """Handles downloading YouTube media or preparing local video files."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or config.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path: Path) -> Path:
        """Extracts 16kHz mono WAV audio track for Whisper transcription."""
        audio_path = self.temp_dir / f"{video_path.stem}_audio.wav"
        if audio_path.exists():
            return audio_path

        cmd = [
            get_ffmpeg_bin(),
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
        return audio_path

    def process_source(self, source: str, resolution: int = 1080) -> Dict[str, Any]:
        """
        Accepts either a YouTube URL or a local file path.
        Returns metadata containing video path, audio path, title, and duration.
        """
        source_path = Path(source)
        if source_path.exists() and source_path.is_file():
            title = source_path.stem
            audio_path = self.extract_audio(source_path)
            return {
                "title": title,
                "video_path": source_path,
                "audio_path": audio_path,
                "is_local": True,
            }

        # Handle YouTube download
        output_template = str(self.temp_dir / "%(title)s_%(id)s.%(ext)s")
        ydl_opts = {
            "format": f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=True)
            title = info.get("title", "downloaded_video")
            filename = ydl.prepare_filename(info)
            # When yt-dlp merges, output might end with .mp4
            video_file = Path(filename).with_suffix(".mp4")
            if not video_file.exists():
                video_file = Path(filename)

        if not video_file.exists():
            raise FileNotFoundError(f"Downloaded video file not found at: {video_file}")

        audio_path = self.extract_audio(video_file)
        return {
            "title": title,
            "video_path": video_file,
            "audio_path": audio_path,
            "duration": info.get("duration", 0),
            "is_local": False,
        }
