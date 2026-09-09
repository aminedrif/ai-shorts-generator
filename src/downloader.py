import re
import json
import glob
import shutil
import subprocess
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import bs4
import yt_dlp
from src.config import config, get_ffmpeg_bin
from src.logger import logger

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


def get_media_duration(video_path: Path) -> float:
    """Calculates exact video duration in seconds via OpenCV, falling back to ffprobe."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frames > 0:
                return round(frames / fps, 2)
    except Exception as e:
        logger.debug(f"OpenCV duration calculation failed: {e}")

    try:
        cmd = [
            get_ffmpeg_bin().replace("ffmpeg", "ffprobe"),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            return round(float(res.stdout.strip()), 2)
    except Exception as pe:
        logger.debug(f"FFprobe duration calculation failed: {pe}")

    return 0.0


def fetch_webpage_html(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches HTML content from an arbitrary webpage using browser headers.
    Returns (html, final_url) or (None, None)."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": BROWSER_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            if any(vt in content_type for vt in ["video/", "application/vnd.apple.mpegurl", "application/x-mpegurl"]):
                return None, resp.geturl()
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            raw = resp.read()
            try:
                return raw.decode(charset, errors="replace"), resp.geturl()
            except Exception:
                return raw.decode("utf-8", errors="replace"), resp.geturl()
    except Exception as e:
        logger.debug(f"Webpage fetch skipped: {e}")
        return None, None


def extract_embedded_video_url(html: str, base_url: str) -> Optional[str]:
    """Inspects arbitrary HTML for video streams, og:video, HTML5 <video>, JSON-LD, or embeds."""
    if not html:
        return None
    try:
        soup = bs4.BeautifulSoup(html, "html.parser")
        for prop in ["og:video", "og:video:url", "og:video:secure_url", "twitter:player:stream"]:
            meta = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
            if meta and meta.get("content"):
                return urllib.parse.urljoin(base_url, meta["content"].strip())

        for v in soup.find_all("video"):
            if v.get("src"):
                return urllib.parse.urljoin(base_url, v["src"].strip())
            for s in v.find_all("source"):
                if s.get("src"):
                    return urllib.parse.urljoin(base_url, s["src"].strip())

        for script in soup.find_all("script", type="application/ld+json"):
            if script.string:
                m = re.search(r'"(?:contentUrl|embedUrl)"\s*:\s*"([^"]+)"', script.string)
                if m:
                    return urllib.parse.urljoin(base_url, m.group(1).strip())

        raw_matches = re.findall(r'https?://[^\s"\'<>]+\.(?:mp4|webm|m3u8|mov|m4v)(?:\?[^\s"\'<>]*)?', html)
        if raw_matches:
            return raw_matches[0].strip()

        for iframe in soup.find_all("iframe"):
            src = iframe.get("src")
            if src and any(p in src for p in ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "player."]):
                return urllib.parse.urljoin(base_url, src.strip())
    except Exception as e:
        logger.debug(f"Failed to parse embedded video from HTML: {e}")

    return None


def download_with_ffmpeg(stream_url: str, output_path: Path) -> Path:
    """Downloads or stream-copies a direct video/HLS URL directly via FFmpeg."""
    ffmpeg_exe = get_ffmpeg_bin()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First attempt: stream copy (lossless, instant)
    cmd_copy = [
        ffmpeg_exe,
        "-y",
        "-user_agent", BROWSER_USER_AGENT,
        "-i", stream_url,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        str(output_path),
    ]
    res = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
        return output_path

    # Fallback attempt: re-encode to standard h264/aac
    cmd_encode = [
        ffmpeg_exe,
        "-y",
        "-user_agent", BROWSER_USER_AGENT,
        "-i", stream_url,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    res_enc = subprocess.run(cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_enc.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg direct download failed: {res_enc.stderr[:300]}")

    return output_path


def sanitize_filename(name: str) -> str:
    """Removes invalid filename characters, quotes, and trims length."""
    clean = re.sub(r'[\\/*?:"<>|\'\`]', "", name)
    clean = re.sub(r"\s+", "_", clean).strip("._")
    return clean[:80] if clean else "video"


def extract_heatmap_peaks(heatmap: Optional[List[Dict[str, Any]]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts the highest viewer replay / retention moments from YouTube's playback heatmap.
    Returns sorted local retention peaks with start, end, and normalized intensity (0.0 - 1.0).
    """
    if not heatmap:
        return []

    sorted_peaks = sorted(heatmap, key=lambda p: p.get("value", 0.0), reverse=True)
    selected_peaks = []

    for item in sorted_peaks:
        s = round(float(item.get("start_time", 0.0)), 2)
        e = round(float(item.get("end_time", 0.0)), 2)
        val = round(float(item.get("value", 0.0)), 3)

        # Ensure selected peaks don't collide closely with each other (minimum 40s separation)
        if not any(abs(s - p["start_time"]) < 40.0 for p in selected_peaks):
            selected_peaks.append({
                "start_time": s,
                "end_time": e,
                "intensity": val,
            })

        if len(selected_peaks) >= top_k:
            break

    return selected_peaks


def parse_json3_to_segments(json3_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a YouTube json3 caption file into timestamped segment dictionaries
    with exact per-word millisecond offsets (tOffsetMs).
    Guarantees sub-frame subtitle synchronization even during fast speech.
    """
    if not json3_path.exists():
        return []

    try:
        data = json.loads(json3_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        logger.warning(f"Failed to parse json3 subtitles: {e}")
        return []

    events = data.get("events", [])
    if not events:
        return []

    all_words = []
    for ev in events:
        t_start = ev.get("tStartMs", 0) / 1000.0
        d_dur = ev.get("dDurationMs", 0) / 1000.0
        segs = ev.get("segs", [])
        if not segs:
            continue

        for i, s in enumerate(segs):
            utf8_text = s.get("utf8", "")
            raw_words = utf8_text.replace("\n", " ").split()
            if not raw_words:
                continue

            offset_s = s.get("tOffsetMs", 0) / 1000.0
            word_start = round(t_start + offset_s, 3)

            if i + 1 < len(segs) and "tOffsetMs" in segs[i + 1]:
                word_end = round(t_start + segs[i + 1]["tOffsetMs"] / 1000.0, 3)
            else:
                word_end = round(min(t_start + d_dur, word_start + 0.35), 3)

            if word_end <= word_start:
                word_end = round(word_start + 0.15, 3)

            if len(raw_words) == 1:
                all_words.append({
                    "word": raw_words[0],
                    "start": word_start,
                    "end": word_end,
                })
            else:
                sub_dur = (word_end - word_start) / len(raw_words)
                for sw_idx, sw in enumerate(raw_words):
                    all_words.append({
                        "word": sw,
                        "start": round(word_start + sw_idx * sub_dur, 3),
                        "end": round(word_start + (sw_idx + 1) * sub_dur, 3),
                    })

    if not all_words:
        return []

    segments = []
    seg_id = 1
    current_words = []

    for w in all_words:
        current_words.append(w)
        is_break = (
            len(current_words) >= 7
            or any(w["word"].endswith(p) for p in [".", "!", "?", ","])
            or (len(current_words) >= 4 and w["end"] - w["start"] > 0.5)
        )
        if is_break:
            seg_s = current_words[0]["start"]
            seg_e = current_words[-1]["end"]
            seg_txt = " ".join([item["word"] for item in current_words])
            segments.append({
                "id": seg_id,
                "start": round(seg_s, 2),
                "end": round(seg_e, 2),
                "text": seg_txt,
                "words": current_words,
            })
            seg_id += 1
            current_words = []

    if current_words:
        seg_s = current_words[0]["start"]
        seg_e = current_words[-1]["end"]
        seg_txt = " ".join([item["word"] for item in current_words])
        segments.append({
            "id": seg_id,
            "start": round(seg_s, 2),
            "end": round(seg_e, 2),
            "text": seg_txt,
            "words": current_words,
        })

    return segments


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
        Extracts real audience retention heatmap data and sub-frame per-word timestamps (json3).
        Returns metadata, heatmap peaks, and parsed segments in ~1-2 seconds, or None if unavailable.
        """
        try:
            temp_sub_prefix = self.temp_dir / f"yt_subs_{int(time.time())}"
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "writesubtitles": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "json3/vtt",
                "outtmpl": str(temp_sub_prefix) + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 15,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "YouTube Video")
                duration = info.get("duration", 0)
                heatmap_data = info.get("heatmap") or []

            heatmap_peaks = extract_heatmap_peaks(heatmap_data, top_k=5)
            if heatmap_peaks:
                logger.info(
                    f"YouTube Most Replayed: Identified {len(heatmap_peaks)} audience retention spikes (Peak: {int(heatmap_peaks[0]['intensity']*100)}% replay intensity at {heatmap_peaks[0]['start_time']}s)."
                )

            # Check for json3 with precise word offsets first, fallback to vtt
            json3_files = glob.glob(f"{temp_sub_prefix}*.json3")
            vtt_files = glob.glob(f"{temp_sub_prefix}*.vtt")

            segments = []
            if json3_files:
                json3_path = Path(json3_files[0])
                segments = parse_json3_to_segments(json3_path)
                try:
                    json3_path.unlink()
                except OSError:
                    pass
                if segments:
                    logger.info("Fast Ingestion: Loaded sub-frame per-word timestamps via YouTube json3 captions.")

            if not segments and vtt_files:
                vtt_path = Path(vtt_files[0])
                segments = parse_vtt_to_segments(vtt_path)
                try:
                    vtt_path.unlink()
                except OSError:
                    pass

            # Clean up any leftover sub files with this prefix
            for f in glob.glob(f"{temp_sub_prefix}*"):
                try:
                    Path(f).unlink(missing_ok=True)
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
                "heatmap_peaks": heatmap_peaks,
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
        Accepts any video link (YouTube, TikTok, Instagram, Twitter/X, Vimeo, direct MP4/stream,
        or arbitrary webpage containing an embedded video), or a local video file path.
        Returns metadata containing video path, audio path, title, and duration.
        """
        source = source.strip()
        source_path = Path(source)
        if source_path.exists() and source_path.is_file():
            title = source_path.stem
            audio_path = self.extract_audio(source_path)
            duration = get_media_duration(source_path)
            return {
                "title": title,
                "video_path": source_path,
                "audio_path": audio_path,
                "duration": duration,
                "is_local": True,
            }

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
                        logger.info(f"[download] {mb_down:.1f}MB at {speed_str}")
            elif status == "finished":
                logger.info("[download] Video stream download finished. Merging media tracks...")

        ydl_opts = {
            "format": f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/bestvideo+bestaudio/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg_exe,
            "noplaylist": True,
            "socket_timeout": 30,
            "retries": 15,
            "fragment_retries": 15,
            "http_chunk_size": 10485760,
            "continuedl": True,
            "http_headers": {
                "User-Agent": BROWSER_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            "progress_hooks": [download_progress_hook],
            "quiet": True,
            "no_warnings": True,
        }

        download_target_url = source
        video_file: Optional[Path] = None
        title = "video"
        duration = 0.0

        # Attempt 1: yt-dlp download
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(download_target_url, download=True)
                title = sanitize_filename(info.get("title", "downloaded_video"))
                duration = float(info.get("duration") or 0.0)
                filename = ydl.prepare_filename(info)
                cand = Path(filename).with_suffix(".mp4")
                if cand.exists():
                    video_file = cand
                elif Path(filename).exists():
                    video_file = Path(filename)
                else:
                    matches = list(self.temp_dir.glob(f"{Path(filename).stem}*"))
                    if matches:
                        video_file = matches[0]
        except Exception as ydl_err:
            logger.warning(f"yt-dlp download failed on {download_target_url}: {ydl_err}. Initiating deep webpage / direct stream fallback...")

            # Check if this is an arbitrary webpage containing an embedded video
            html, final_url = fetch_webpage_html(download_target_url)
            discovered_video = extract_embedded_video_url(html, final_url or download_target_url) if html else None

            if discovered_video and discovered_video != download_target_url:
                logger.info(f"Discovered embedded video stream in webpage: {discovered_video}")
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(discovered_video, download=True)
                        title = sanitize_filename(info.get("title", "webpage_video"))
                        duration = float(info.get("duration") or 0.0)
                        filename = ydl.prepare_filename(info)
                        cand = Path(filename).with_suffix(".mp4")
                        if cand.exists():
                            video_file = cand
                        elif Path(filename).exists():
                            video_file = Path(filename)
                except Exception as ydl_inner_err:
                    logger.warning(f"yt-dlp failed on discovered video: {ydl_inner_err}. Trying direct FFmpeg stream copy...")
                    fallback_out = self.temp_dir / f"stream_{int(time.time())}.mp4"
                    video_file = download_with_ffmpeg(discovered_video, fallback_out)
                    title = "webpage_video"
            else:
                # Direct FFmpeg stream copy fallback on original URL
                logger.info(f"Trying direct FFmpeg stream download on {download_target_url}...")
                fallback_out = self.temp_dir / f"direct_{int(time.time())}.mp4"
                video_file = download_with_ffmpeg(download_target_url, fallback_out)
                title = sanitize_filename(Path(urllib.parse.urlparse(download_target_url).path).stem or "direct_video")

        if not video_file or not video_file.exists():
            raise FileNotFoundError(f"Could not download or extract video from: {source}")

        if duration <= 0:
            duration = get_media_duration(video_file)

        audio_path = self.extract_audio(video_file)
        return {
            "title": title,
            "video_path": video_file,
            "audio_path": audio_path,
            "duration": duration,
            "is_local": False,
        }

    def extract_preview_info(self, url: str) -> Dict[str, Any]:
        """
        Fast inspection of any link to determine platform, title, thumbnail, duration,
        and playable stream URL for the iPhone 16 live mockup.
        """
        url = url.strip()
        if not url:
            return {"status": "empty", "platform": "none", "platform_label": ""}

        # 1. YouTube check
        yt_match = re.search(r'(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|shorts\/|live\/|watch\?.+&v=))([\w-]{11})', url)
        if yt_match:
            yt_id = yt_match.group(1)
            return {
                "status": "ok",
                "platform": "youtube",
                "platform_label": "YouTube",
                "video_id": yt_id,
                "thumbnail": f"https://img.youtube.com/vi/{yt_id}/maxresdefault.jpg",
                "title": "YouTube Video",
                "direct_video_url": None,
            }

        # 2. Direct video file check
        parsed_path = urllib.parse.urlparse(url).path.lower()
        if any(parsed_path.endswith(ext) for ext in [".mp4", ".webm", ".mov", ".m4v", ".m3u8", ".mpd", ".ogg"]):
            stem = Path(parsed_path).stem.replace("_", " ").replace("-", " ").title()
            return {
                "status": "ok",
                "platform": "direct",
                "platform_label": "Direct Video",
                "title": stem or "Direct Video Stream",
                "direct_video_url": url,
                "thumbnail": None,
            }

        # 3. Fast inspection with yt-dlp
        try:
            ydl_opts = {
                "skip_download": True,
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 8,
                "http_headers": {"User-Agent": BROWSER_USER_AGENT},
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                extractor = (info.get("extractor_key") or "web").lower()
                title = info.get("title") or "Video"
                thumb = info.get("thumbnail")
                duration = info.get("duration") or 0.0

                label = "Web Video"
                if "tiktok" in extractor:
                    label = "TikTok"
                elif "instagram" in extractor:
                    label = "Instagram Reel"
                elif "twitter" in extractor or "x" in extractor:
                    label = "X / Twitter"
                elif "vimeo" in extractor:
                    label = "Vimeo"
                elif "reddit" in extractor:
                    label = "Reddit"
                elif "facebook" in extractor:
                    label = "Facebook"
                elif "twitch" in extractor:
                    label = "Twitch"

                direct_url = None
                formats = info.get("formats") or []
                for f in reversed(formats):
                    f_url = f.get("url")
                    f_ext = (f.get("ext") or "").lower()
                    vcodec = f.get("vcodec") or ""
                    if f_url and f_ext in ["mp4", "webm"] and vcodec != "none" and "http" in f_url:
                        direct_url = f_url
                        break
                if not direct_url and info.get("url") and any(ext in info.get("url") for ext in [".mp4", ".webm", ".m3u8"]):
                    direct_url = info.get("url")

                return {
                    "status": "ok",
                    "platform": extractor,
                    "platform_label": label,
                    "title": title,
                    "thumbnail": thumb,
                    "duration": duration,
                    "direct_video_url": direct_url,
                }
        except Exception as e:
            logger.debug(f"yt-dlp preview extraction skipped: {e}")

        # 4. Arbitrary webpage inspection
        html, final_url = fetch_webpage_html(url)
        if html:
            embedded_url = extract_embedded_video_url(html, final_url or url)
            soup = bs4.BeautifulSoup(html, "html.parser")
            page_title = soup.title.string.strip() if (soup.title and soup.title.string) else "Webpage Video"
            og_img = None
            meta_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
            if meta_img and meta_img.get("content"):
                og_img = urllib.parse.urljoin(final_url or url, meta_img["content"])

            if embedded_url:
                return {
                    "status": "ok",
                    "platform": "webpage",
                    "platform_label": "Webpage Video",
                    "title": page_title,
                    "thumbnail": og_img,
                    "direct_video_url": embedded_url if any(ext in embedded_url for ext in [".mp4", ".webm", ".mov"]) else None,
                }

        return {
            "status": "ok",
            "platform": "link",
            "platform_label": "Web Link",
            "title": "Video Link",
            "thumbnail": None,
            "direct_video_url": None,
        }
