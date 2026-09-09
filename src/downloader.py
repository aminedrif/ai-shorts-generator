import re
import glob
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import yt_dlp
from src.config import config, get_ffmpeg_bin
from src.logger import logger


def sanitize_filename(name: str) -> str:
    """Removes invalid filename characters, quotes, and trims length."""
    clean = re.sub(r'[\\/*?:"<>|\'\`]', "", name)
    clean = re.sub(r"\s+", "_", clean).strip("._")
    return clean[:80] if clean else "video"


def parse_vtt_to_segments(vtt_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a WebVTT caption file into standard timestamped segment dictionaries.
    """
    if not vtt_path.exists():
        return []

    lines = vtt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    segments: List[Dict[str, Any]] = []
    time_pattern = re.compile(
        r"^(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})"
    )

    def to_secs(h, m, s, ms):
        hours = int(h) if h else 0
        return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    idx = 0
    seg_id = 1
    while idx < len(lines):
        line = lines[idx].strip()
        match = time_pattern.match(line)
        if match:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
            start_sec = to_secs(h1, m1, s1, ms1)
            end_sec = to_secs(h2, m2, s2, ms2)

            idx += 1
            text_lines = []
            while idx < len(lines) and lines[idx].strip() and not time_pattern.match(lines[idx].strip()):
                clean_text = re.sub(r"<[^>]+>", "", lines[idx].strip())
                if clean_text:
                    text_lines.append(clean_text)
                idx += 1

            full_text = " ".join(text_lines).strip()
            if full_text and (not segments or segments[-1]["text"] != full_text):
                words = full_text.split()
                w_count = max(1, len(words))
                w_dur = (end_sec - start_sec) / w_count
                word_objects = []
                for w_i, word in enumerate(words):
                    w_s = round(start_sec + w_i * w_dur, 2)
                    w_e = round(w_s + w_dur, 2)
                    word_objects.append({"word": word, "start": w_s, "end": w_e})

                segments.append({
                    "id": seg_id,
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "text": full_text,
                    "words": word_objects,
                })
                seg_id += 1
        else:
            idx += 1

    return segments


class VideoDownloader:
    """Handles downloading YouTube media, fetching captions, or preparing local video files."""

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

    def fetch_youtube_transcript(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to fetch YouTube's built-in or auto-generated subtitles without downloading any media.
        Returns metadata and parsed segments in ~1-2 seconds, or None if unavailable.
        """
        try:
            temp_sub_prefix = self.temp_dir / f"yt_subs_{int(time.time())}"
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "writesubtitles": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "vtt",
                "outtmpl": str(temp_sub_prefix) + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 15,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "YouTube Video")
                duration = info.get("duration", 0)

            vtt_files = glob.glob(f"{temp_sub_prefix}*.vtt")
            if not vtt_files:
                return None

            vtt_path = Path(vtt_files[0])
            segments = parse_vtt_to_segments(vtt_path)
            try:
                vtt_path.unlink()
            except OSError:
                pass

            if not segments:
                return None

            full_text = " ".join([s["text"] for s in segments])
            logger.info(f"Fast Ingestion: Retrieved {len(segments)} transcript cues from YouTube in seconds.")
            return {
                "title": title,
                "duration": duration,
                "segments": segments,
                "full_text": full_text,
            }
        except Exception as e:
            logger.warning(f"YouTube transcript extraction skipped ({e}); falling back to standard media download.")
            return None

    def download_video_section(
        self,
        url: str,
        start_time: float,
        end_time: float,
        output_path: Path,
        resolution: int = 1080,
    ) -> Path:
        """
        Downloads ONLY a specific time slice [start_time, end_time] of a YouTube video using HTTP range requests.
        Downloads ~15-25 MB in seconds instead of multi-gigabyte full videos.
        """
        ffmpeg_exe = get_ffmpeg_bin()
        ffmpeg_dir = str(Path(ffmpeg_exe).parent)

        def section_range(info_dict, ydl):
            return [{
                "start_time": max(0.0, float(start_time)),
                "end_time": float(end_time),
                "title": output_path.stem,
            }]

        ydl_opts = {
            "format": f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/best",
            "outtmpl": str(output_path),
            "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg_dir,
            "download_ranges": section_range,
            "force_keyframes_at_cuts": True,
            "noplaylist": True,
            "socket_timeout": 30,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not output_path.exists():
            candidates = list(output_path.parent.glob(f"{output_path.stem}*"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(f"Section download failed to create {output_path}")

        return output_path

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
        ffmpeg_exe = get_ffmpeg_bin()
        output_template = str(self.temp_dir / "%(title)s_%(id)s.%(ext)s")

        last_log_time = 0.0

        def download_progress_hook(d):
            nonlocal last_log_time
            status = d.get("status")
            if status == "downloading":
                now = time.time()
                if now - last_log_time >= 3.0:
                    last_log_time = now
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded = d.get("downloaded_bytes", 0)
                    speed = d.get("speed") or 0
                    speed_str = f"{speed / (1024 * 1024):.1f} MB/s" if speed else "calculating..."
                    if total > 0:
                        pct = (downloaded / total) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_tot = total / (1024 * 1024)
                        logger.info(f"[download] {pct:.1f}% ({mb_down:.0f}MB / {mb_tot:.0f}MB) at {speed_str}")
                    else:
                        mb_down = downloaded / (1024 * 1024)
                        mb_tot = total / (1024 * 1024) if total else 0
                        logger.info(f"[download] {mb_down:.1f}MB at {speed_str}")
            elif status == "finished":
                logger.info("[download] Stream download complete. Merging streams...")

        ydl_opts = {
            "format": f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg_exe,
            "noplaylist": True,
            "socket_timeout": 30,
            "retries": 15,
            "fragment_retries": 15,
            "http_chunk_size": 10485760,
            "continuedl": True,
            "progress_hooks": [download_progress_hook],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=True)
            title = info.get("title", "downloaded_video")
            filename = ydl.prepare_filename(info)
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
