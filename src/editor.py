import os
import sys
import re
import time
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.config import config, get_ffmpeg_bin, is_nvenc_available
from src.downloader import sanitize_filename, get_media_duration
from src.logger import logger, track_error


def clean_caption_text(text: str) -> str:
    """Removes >>, &, gt, HTML artifacts, and emoji characters from text."""
    if not text:
        return ""
    cleaned = re.sub(r"&gt;&gt;|&gt;|>>|>{1,}|«|»", " ", text)
    cleaned = re.sub(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\U0001f300-\U0001f9ff]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

def contains_arabic_script(text: str) -> bool:
    """Detects if string contains Arabic or Perso-Arabic unicode characters."""
    if not text:
        return False
    return any(
        '\u0600' <= ch <= '\u06FF' or
        '\u0750' <= ch <= '\u077F' or
        '\u08A0' <= ch <= '\u08FF' or
        '\uFB50' <= ch <= '\uFDFF' or
        '\uFE70' <= ch <= '\uFEFF'
        for ch in text
    )


def format_arabic_for_subtitles(text: str) -> str:
    """
    Returns clean text for FFmpeg libass subtitles.
    Modern FFmpeg libass contains HarfBuzz and FriBidi, so raw Unicode Arabic strings
    are shaped and reordered automatically. Pre-reshaping or BiDi reordering causes
    double-inversion and disconnected characters.
    """
    if not text:
        return ""
    return text.strip()


def format_srt_time(seconds: float) -> str:
    """Converts seconds into SRT timestamp format: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_ass_time(seconds: float) -> str:
    """Converts seconds into ASS timestamp format: H:MM:SS.cc"""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


_GENDER_CACHE: Dict[str, str] = {}


def hex_to_ass_color(hex_str: str, default: str = "&H0015CCFA") -> str:
    """Converts a standard #RRGGBB hex color to ASS &H00BBGGRR format."""
    if not hex_str:
        return default
    clean = hex_str.strip()
    if clean.startswith("&H"):
        return clean
    clean = clean.lstrip("#")
    if len(clean) == 6:
        r = clean[0:2]
        g = clean[2:4]
        b = clean[4:6]
        return f"&H00{b.upper()}{g.upper()}{r.upper()}"
    return default


def detect_voice_gender(video_or_audio_path: Optional[Path], timestamp: float, duration: float = 1.0) -> str:
    """
    Analyzes audio pitch around timestamp to detect male vs female voice.
    Returns '👨' for male voice (< 165 Hz), '👩' for female voice (>= 165 Hz).
    """
    if not video_or_audio_path or not Path(video_or_audio_path).exists():
        return "👨"

    cache_key = f"{Path(video_or_audio_path).name}_{int(timestamp)}"
    if cache_key in _GENDER_CACHE:
        return _GENDER_CACHE[cache_key]

    try:
        import numpy as np

        ffmpeg_bin = get_ffmpeg_bin()
        cmd = [
            ffmpeg_bin,
            "-ss", str(max(0.0, float(timestamp))),
            "-t", str(float(duration)),
            "-i", str(video_or_audio_path),
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-loglevel", "quiet",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=4)
        raw_bytes = proc.stdout
        if len(raw_bytes) < 3200:
            return "👨"

        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        samples -= np.mean(samples)
        std = np.std(samples)
        if std < 10.0:
            return "👨"
        samples /= std

        corr = np.correlate(samples, samples, mode="full")
        corr = corr[len(corr) // 2:]

        sample_rate = 16000
        min_lag = int(sample_rate / 320)
        max_lag = int(sample_rate / 75)
        if max_lag >= len(corr):
            return "👨"

        peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        f0 = sample_rate / peak_lag

        result = "👨" if f0 < 165.0 else "👩"
        _GENDER_CACHE[cache_key] = result
        return result
    except Exception as e:
        logger.debug(f"Pitch gender estimation fallback: {e}")
        return "👨"


def get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """Returns (width, height) of the video file using OpenCV, falling back to 1920x1080."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    return 1920, 1080


def get_cascade_paths() -> Dict[str, str]:
    """Resolves file paths to bundled Haar cascades."""
    candidate_dirs = [
        Path(__file__).resolve().parent / "cascades",
        config.base_dir / "src" / "cascades",
        config.base_dir / "cascades",
    ]
    if hasattr(sys, "_MEIPASS"):
        candidate_dirs.append(Path(sys._MEIPASS) / "src" / "cascades")
        candidate_dirs.append(Path(sys._MEIPASS) / "cascades")

    for d in candidate_dirs:
        alt2 = d / "haarcascade_frontalface_alt2.xml"
        if alt2.is_file():
            return {
                "alt2": str(alt2),
                "prof": str(d / "haarcascade_profileface.xml"),
                "default": str(d / "haarcascade_frontalface_default.xml"),
            }

    try:
        import cv2
        cv2_data = getattr(cv2, "data", None)
        if cv2_data and hasattr(cv2_data, "haarcascades"):
            base = cv2_data.haarcascades
            return {
                "alt2": base + "haarcascade_frontalface_alt2.xml",
                "prof": base + "haarcascade_profileface.xml",
                "default": base + "haarcascade_frontalface_default.xml",
            }
    except Exception:
        pass
    return {}


def detect_streamer_facecam_box(
    video_path: Path,
    start_time: float,
    duration: float,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the 2D bounding box (x, y, w, h) of a streamer's facecam box
    in original video coordinates across sampled frames.
    """
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_h <= 0 or orig_w <= 0:
            cap.release()
            return None

        cascades = get_cascade_paths()
        if not cascades:
            cap.release()
            return None

        clf_alt2 = cv2.CascadeClassifier(cascades.get("alt2", ""))
        clf_prof = cv2.CascadeClassifier(cascades.get("prof", ""))
        clf_def = cv2.CascadeClassifier(cascades.get("default", ""))

        sample_points = np.linspace(start_time, start_time + max(0.5, duration), 15)
        detected_boxes = []

        for sec in sample_points:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = clf_alt2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces) == 0 and not clf_prof.empty():
                faces = clf_prof.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces) == 0 and not clf_def.empty():
                faces = clf_def.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            if len(faces) > 0:
                largest = max(faces, key=lambda b: b[2] * b[3])
                detected_boxes.append((int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3])))

        cap.release()

        if not detected_boxes:
            return None

        med_x = int(np.median([b[0] for b in detected_boxes]))
        med_y = int(np.median([b[1] for b in detected_boxes]))
        med_w = int(np.median([b[2] for b in detected_boxes]))
        med_h = int(np.median([b[3] for b in detected_boxes]))

        return (med_x, med_y, med_w, med_h)
    except Exception as e:
        logger.warning(f"Streamer facecam detection error: {e}")
        return None


def build_smooth_pan_expression(
    detections: List[Tuple[float, int]],
    duration: float,
    max_crop: int,
    default_crop: int,
) -> str:
    """
    Builds a continuous, smooth FFmpeg time-based crop expression (piecewise linear with ease transitions)
    that dynamically follows the active speaker without jarring jitter.
    """
    import numpy as np

    if not detections:
        return str(default_crop)

    # Group detections into 2.0-second temporal windows
    window_sec = 2.0
    num_windows = max(1, int(np.ceil(duration / window_sec)))

    window_targets = []
    for w in range(num_windows):
        t_start = w * window_sec
        t_end = min(duration, (w + 1) * window_sec)
        t_mid = (t_start + t_end) / 2.0
        pts = [c for t, c in detections if t_start <= t < t_end]
        if pts:
            window_targets.append((t_mid, int(np.median(pts))))
        elif window_targets:
            window_targets.append((t_mid, window_targets[-1][1]))
        else:
            window_targets.append((t_mid, default_crop))

    # Apply camera deadband: ignore movements < 70 pixels
    filtered_targets = [window_targets[0]]
    for t_mid, x in window_targets[1:]:
        prev_x = filtered_targets[-1][1]
        if abs(x - prev_x) < 70:
            filtered_targets.append((t_mid, prev_x))
        else:
            filtered_targets.append((t_mid, x))

    # If all windows point to the same target, return clean constant
    unique_xs = set(x for _, x in filtered_targets)
    if len(unique_xs) == 1:
        return str(list(unique_xs)[0])

    # Construct nested FFmpeg if expression with smooth ease transitions
    expr = str(filtered_targets[-1][1])
    for i in range(len(filtered_targets) - 2, -1, -1):
        t1, x1 = filtered_targets[i]
        t2, x2 = filtered_targets[i + 1]
        if x1 == x2:
            expr = f"if(lt(t,{t2:.2f}),{x1},{expr})"
        else:
            trans_dur = min(1.0, max(0.5, (t2 - t1) * 0.4))
            t_trans_start = max(0.0, t2 - trans_dur)
            lerp = f"{x1}+({x2}-{x1})*(t-{t_trans_start:.2f})/{trans_dur:.2f}"
            expr = f"if(lt(t,{t_trans_start:.2f}),{x1},if(lt(t,{t2:.2f}),{lerp},{expr}))"

    return f"min({max_crop},max(0,{expr}))"


def detect_dynamic_speaker_crop_expression(
    video_path: Path,
    start_time: float,
    duration: float,
    target_w: int = 1080,
    target_h: int = 1920,
) -> str:
    """
    Analyzes video frames across the clip using bundled multi-angle face cascades
    and generates a dynamic, camera-following FFmpeg crop expression centered on the speaker.
    """
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return f"(in_w-{target_w})/2"

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if orig_h <= 0 or orig_w <= 0:
            return f"(in_w-{target_w})/2"

        scale = target_h / orig_h
        scaled_w = int(orig_w * scale)
        if scaled_w <= target_w:
            return "0"

        max_crop_x = scaled_w - target_w
        default_crop_x = max_crop_x // 2

        cascades = get_cascade_paths()
        if not cascades:
            return f"{default_crop_x}"

        clf_alt2 = cv2.CascadeClassifier(cascades.get("alt2", ""))
        clf_prof = cv2.CascadeClassifier(cascades.get("prof", ""))
        clf_def = cv2.CascadeClassifier(cascades.get("default", ""))

        # Fast frame sampling using FFmpeg pipe
        sample_fps = 2
        thumb_w = 360
        thumb_h = int(thumb_w * orig_h / orig_w)
        frame_bytes_len = thumb_w * thumb_h * 3

        cmd = [
            get_ffmpeg_bin(),
            "-y",
            "-ss", f"{start_time:.2f}",
            "-i", str(video_path),
            "-t", f"{duration:.2f}",
            "-vf", f"fps={sample_fps},scale={thumb_w}:{thumb_h}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        raw_data, _ = proc.communicate(timeout=15)

        num_frames = len(raw_data) // frame_bytes_len
        if num_frames == 0:
            return f"{default_crop_x}"

        detections: List[Tuple[float, int]] = []

        for i in range(num_frames):
            t_rel = round(i / sample_fps, 2)
            chunk = raw_data[i * frame_bytes_len : (i + 1) * frame_bytes_len]
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((thumb_h, thumb_w, 3))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # 1. Frontal alt2 (high accuracy)
            faces = clf_alt2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            if len(faces) == 0 and not clf_prof.empty():
                # 2. Profile right
                faces = clf_prof.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                if len(faces) == 0:
                    # 3. Profile left (flipped)
                    flipped = cv2.flip(gray, 1)
                    ffaces = clf_prof.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                    if len(ffaces) > 0:
                        faces = np.array([[thumb_w - f[0] - f[2], f[1], f[2], f[3]] for f in ffaces])

            if len(faces) == 0 and not clf_def.empty():
                faces = clf_def.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

            if len(faces) > 0:
                valid_faces = [f for f in faces if (f[1] + f[3] / 2) < thumb_h * 0.85]
                chosen = valid_faces if valid_faces else faces
                best = max(chosen, key=lambda f: f[2] * f[3] * (1.2 if (f[1] + f[3] / 2) < thumb_h * 0.6 else 1.0))

                face_center_x = (best[0] + best[2] / 2.0) * (orig_w / thumb_w)
                scaled_x = face_center_x * scale
                target_crop = int(scaled_x - target_w / 2.0)
                clamped_crop = max(0, min(max_crop_x, target_crop))
                detections.append((t_rel, clamped_crop))

        if not detections:
            logger.info("Face tracking: no clear speaker face detected, using center framing.")
            return f"{default_crop_x}"

        crops = [d[1] for d in detections]
        spread = max(crops) - min(crops)

        if spread < 80:
            median_crop = int(np.median(crops))
            logger.info(f"Smart face-tracking: stationary speaker centered at crop_x={median_crop} (center was {default_crop_x})")
            return f"{median_crop}"

        logger.info(f"Smart face-tracking: dynamic camera tracking activated (spread={spread}px across {len(detections)} frames)")
        return build_smooth_pan_expression(detections, duration, max_crop_x, default_crop_x)

    except Exception as e:
        logger.warning(f"Smart face tracking error, defaulting to center crop: {e}")
        return f"(in_w-{target_w})/2"


def detect_speaker_crop_x(
    video_path: Path,
    start_time: float,
    duration: float,
    target_w: int = 1080,
    target_h: int = 1920,
) -> int:
    """Legacy helper returning static integer crop offset."""
    expr = detect_dynamic_speaker_crop_expression(video_path, start_time, duration, target_w, target_h)
    try:
        return int(expr)
    except ValueError:
        return (1080 - target_w) // 2


class VideoEditor:
    """Handles video cutting, 9:16 vertical re-framing, and subtitle burning using FFmpeg."""

    def __init__(self, output_dir: Optional[Path] = None, temp_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.output_dir
        self.temp_dir = temp_dir or config.temp_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def generate_subtitles_srt(
        self,
        transcript_segments: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
        output_path: Path,
        video_path: Optional[Path] = None,
    ) -> Path:
        """Generates a standard SRT subtitle file shifted to clip-relative timestamps with no overlaps and no emojis."""
        srt_lines = []
        counter = 1
        last_end = 0.0

        for seg in transcript_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            if seg_end <= start_time or seg_start >= end_time:
                continue

            rel_start = max(last_end, max(0.0, seg_start - start_time))
            rel_end = min(end_time - start_time, seg_end - start_time)

            if rel_end - rel_start < 0.1:
                continue

            text = clean_caption_text(seg["text"]).strip()
            if not text:
                continue

            text = format_arabic_for_subtitles(text)
            srt_lines.append(str(counter))
            srt_lines.append(f"{format_srt_time(rel_start)} --> {format_srt_time(rel_end)}")
            srt_lines.append(text)
            srt_lines.append("")
            counter += 1
            last_end = rel_end

        output_path.write_text("\n".join(srt_lines), encoding="utf-8")
        return output_path

    def generate_subtitles_ass(
        self,
        transcript_segments: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
        output_path: Path,
        hook_title: Optional[str] = None,
        karaoke: bool = True,
        video_path: Optional[Path] = None,
        highlight_color: str = "#facc15",
        font_name: str = "Montserrat",
        font_size: int = 64,
        margin_v: int = 280,
    ) -> Path:
        """
        Generates an ASS subtitle file with animated karaoke active-word highlighting,
        custom highlight color, speaker turn emoji detection (👨 / 👩), and optional hook title banner in the top safe zone.
        """
        duration = end_time - start_time
        ass_highlight = hex_to_ass_color(highlight_color, default="&H0015CCFA")
        clean_font = font_name.strip() if font_name else "Montserrat"
        font_size = int(font_size) if font_size else 64
        margin_v = int(margin_v) if margin_v else 280

        ass_header = [
            "[Script Info]",
            "Title: AI Shorts Captions",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: None",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: KaraokeSub,{clean_font},{font_size},&H00FFFFFF,{ass_highlight},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,{margin_v},1",
            f"Style: HookTitle,{clean_font},48,&H00FFFFFF,&H00000000,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,3,12,0,8,60,60,180,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        events = []

        # 1. Hook title banner in top safe area for first ~3.5 seconds
        if hook_title and hook_title.strip():
            banner_end = min(3.5, max(1.5, duration))
            raw_hook = hook_title.replace("\n", " ").strip()
            clean_hook = raw_hook if contains_arabic_script(raw_hook) else raw_hook.upper()
            if clean_hook:
                events.append(
                    f"Dialogue: 1,0:00:00.20,{format_ass_time(banner_end)},HookTitle,,0,0,0,,{{\\fad(200,400)}}{clean_hook}"
                )

        # 2. Extract words falling inside [start_time, end_time] without emojis or >>
        valid_words = []
        for seg in transcript_segments:
            for w in seg.get("words", []):
                w_start = float(w["start"])
                w_end = float(w["end"])
                if w_end > start_time and w_start < end_time:
                    rel_s = max(0.0, w_start - start_time)
                    rel_e = min(duration, w_end - start_time)
                    if rel_e > rel_s:
                        clean_w = clean_caption_text(w["word"]).strip()
                        if clean_w:
                            disp_w = clean_w if contains_arabic_script(clean_w) else clean_w.upper()
                            valid_words.append({
                                "word": disp_w,
                                "start": rel_s,
                                "end": rel_e,
                            })

        # Strict deduplication and monotonic non-overlapping word sequencing
        if valid_words:
            valid_words.sort(key=lambda x: x["start"])
            deduped = []
            for w in valid_words:
                if deduped:
                    prev = deduped[-1]
                    # Discard identical word repetitions within 0.4s (common in rolling auto-captions)
                    if prev["word"] == w["word"] and abs(w["start"] - prev["start"]) < 0.4:
                        continue
                    if w["start"] < prev["start"]:
                        continue
                    if prev["end"] > w["start"]:
                        prev["end"] = max(prev["start"] + 0.08, w["start"])
                deduped.append(w)
            valid_words = deduped

        if karaoke and valid_words:
            # Group words into natural phrases of 3-5 words or punctuation breaks
            chunks = []
            current_chunk = []
            for w in valid_words:
                current_chunk.append(w)
                has_punct = any(w["word"].endswith(p) for p in [".", "!", "?", ",", "،", "؟"])
                if len(current_chunk) >= 5 or (len(current_chunk) >= 3 and has_punct):
                    chunks.append(current_chunk)
                    current_chunk = []
            if current_chunk:
                chunks.append(current_chunk)

            for chunk_idx, chunk in enumerate(chunks):
                c_start = chunk[0]["start"]
                last_w_end = chunk[-1]["end"]

                # Ensure chunk end strictly finishes BEFORE the next chunk starts to prevent overlapping text
                if chunk_idx + 1 < len(chunks):
                    next_c_start = chunks[chunk_idx + 1][0]["start"]
                    c_end = min(last_w_end + 0.25, max(c_start + 0.12, next_c_start - 0.03))
                else:
                    c_end = min(duration, last_w_end + 0.5)

                if c_end <= c_start:
                    c_end = c_start + 0.12

                chunk_has_arabic = any(contains_arabic_script(w["word"]) for w in chunk)
                if chunk_has_arabic:
                    phrase_text = " ".join(w["word"] for w in chunk)
                    events.append(
                        f"Dialogue: 0,{format_ass_time(c_start)},{format_ass_time(c_end)},KaraokeSub,,0,0,0,,{{\\c{ass_highlight}}}{phrase_text}"
                    )
                else:
                    # For Latin scripts, active word pop karaoke with strictly non-overlapping event timings
                    for i, active_w in enumerate(chunk):
                        w_start = active_w["start"]
                        if i + 1 < len(chunk):
                            next_w_start = chunk[i + 1]["start"]
                            w_end = min(next_w_start, c_end)
                            if w_end <= w_start:
                                w_end = w_start + 0.08
                        else:
                            w_end = c_end

                        if w_end <= w_start:
                            w_end = w_start + 0.08

                        formatted_parts = []
                        for j, w in enumerate(chunk):
                            if j < i:
                                formatted_parts.append(f"{{\\c&H00D0D0D0}}{w['word']}")
                            elif j == i:
                                formatted_parts.append(f"{{\\c{ass_highlight}\\t(0,60,\\fscx106\\fscy106)}}{w['word']}")
                            else:
                                formatted_parts.append(f"{{\\c&H00FFFFFF}}{w['word']}")

                        line_text = " ".join(formatted_parts)
                        events.append(
                            f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},KaraokeSub,,0,0,0,,{line_text}"
                        )
        else:
            # Fallback to segment-level subtitles with zero overlap
            last_event_end = 0.0
            for seg in transcript_segments:
                seg_s = float(seg["start"])
                seg_e = float(seg["end"])
                if seg_e > start_time and seg_s < end_time:
                    rel_s = max(last_event_end, max(0.0, seg_s - start_time))
                    rel_e = min(duration, seg_e - start_time)
                    if rel_e <= rel_s:
                        continue
                    clean_txt = clean_caption_text(seg["text"]).strip()
                    text = clean_txt if contains_arabic_script(clean_txt) else clean_txt.upper()
                    if text:
                        events.append(
                            f"Dialogue: 0,{format_ass_time(rel_s)},{format_ass_time(rel_e)},KaraokeSub,,0,0,0,,{{\\c&H00FFFFFF}}{text}"
                        )
                        last_event_end = rel_e

        full_content = "\n".join(ass_header + events)
        output_path.write_text(full_content, encoding="utf-8")
        return output_path

    def render_clip(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        title: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
        ratio: str = "9:16",
        burn_subtitles: bool = True,
        style: str = "smart_crop",
        hook_title: Optional[str] = None,
        subtitle_style: str = "karaoke",
        is_precut: bool = False,
        highlight_color: str = "#facc15",
        font_name: str = "Montserrat",
        font_size: int = 64,
    ) -> Path:
        """
        Cuts, scales, crops, and burns subtitles for a single short clip.
        Styles: 'smart_crop' (face tracking), 'blurred_background', or 'crop_center'.
        """
        clean_title = sanitize_filename(title)
        output_filename = f"{clean_title}_{int(start_time)}_{int(end_time)}.mp4"
        output_path = self.output_dir / output_filename
        duration = end_time - start_time

        # Prepare dimensions for 9:16 (1080x1920) or 1:1 (1080x1080)
        target_w, target_h = 1080, 1920
        if ratio == "1:1":
            target_w, target_h = 1080, 1080

        filter_complex = []
        crop_sample_start = 0.0 if is_precut else start_time
        orig_w, orig_h = get_video_dimensions(video_path)

        if style in ("split_screen", "stream_split"):
            # Split-Screen Stream Layout:
            # Top half (1080x960): Streamer facecam zoomed in
            # Bottom half (1080x960): Gameplay/screen content
            half_h = target_h // 2
            face_box = detect_streamer_facecam_box(video_path, crop_sample_start, duration)
            if face_box:
                fx, fy, fw, fh = face_box
                cam_h = min(orig_h, max(int(fh * 3.2), int(orig_h * 0.45)))
                cam_w = int(cam_h * (target_w / half_h))
                if cam_w > orig_w:
                    cam_w = orig_w
                    cam_h = int(cam_w * (half_h / target_w))
                cam_x = max(0, min(orig_w - cam_w, int(fx + fw / 2 - cam_w / 2)))
                cam_y = max(0, min(orig_h - cam_h, int(fy + fh * 0.4 - cam_h * 0.45)))
            else:
                cam_w = int(orig_h * 0.5 * (target_w / half_h))
                cam_h = int(orig_h * 0.5)
                cam_x = 0
                cam_y = 0

            # Gameplay: centered 9:8
            game_h = orig_h
            game_w = int(game_h * (target_w / half_h))
            if game_w > orig_w:
                game_w = orig_w
                game_h = int(game_w * (half_h / target_w))
            game_x = max(0, (orig_w - game_w) // 2)
            game_y = max(0, (orig_h - game_h) // 2)

            filter_complex.append(
                f"[0:v]split=2[cam_in][game_in];"
                f"[cam_in]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},"
                f"scale={target_w}:{half_h}:flags=lanczos[top_half];"
                f"[game_in]crop={game_w}:{game_h}:{game_x}:{game_y},"
                f"scale={target_w}:{half_h}:flags=lanczos[bot_half];"
                f"[top_half][bot_half]vstack=inputs=2[vcomposed]"
            )
            current_video_label = "[vcomposed]"
        elif style in ("stream_focus", "speaker_focus"):
            # Focus on streamer facecam and ignore messages/chat
            face_box = detect_streamer_facecam_box(video_path, crop_sample_start, duration)
            if face_box:
                fx, fy, fw, fh = face_box
                ch = min(orig_h, max(int(fh * 3.6), int(orig_h * 0.65)))
                cw = int(ch * (target_w / target_h))
                if cw > orig_w:
                    cw = orig_w
                    ch = int(cw * (target_h / target_w))
                cx = max(0, min(orig_w - cw, int(fx + fw / 2 - cw / 2)))
                cy = max(0, min(orig_h - ch, int(fy + fh * 0.4 - ch * 0.4)))
                filter_complex.append(
                    f"[0:v]crop={cw}:{ch}:{cx}:{cy},"
                    f"scale={target_w}:{target_h}:flags=lanczos[vcomposed]"
                )
            else:
                crop_expr = detect_dynamic_speaker_crop_expression(video_path, crop_sample_start, duration, target_w, target_h)
                filter_complex.append(
                    f"[0:v]scale=-1:{target_h}:flags=lanczos,"
                    f"crop={target_w}:{target_h}:x=\'{crop_expr}\':y=0[vcomposed]"
                )
            current_video_label = "[vcomposed]"
        elif style == "blurred_background":
            filter_complex.append(
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={target_w}:{target_h},boxblur=20:5[bgblurred];"
                f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos[fgscaled];"
                f"[bgblurred][fgscaled]overlay=(W-w)/2:(H-h)/2[vcomposed]"
            )
            current_video_label = "[vcomposed]"
        elif style in ("smart_crop", "crop_smart"):
            crop_expr = detect_dynamic_speaker_crop_expression(video_path, crop_sample_start, duration, target_w, target_h)
            filter_complex.append(
                f"[0:v]scale=-1:{target_h}:flags=lanczos,"
                f"crop={target_w}:{target_h}:x=\'{crop_expr}\':y=0[vcomposed]"
            )
            current_video_label = "[vcomposed]"
        else:
            # Standard center crop
            filter_complex.append(
                f"[0:v]scale=-1:{target_h}:flags=lanczos,"
                f"crop={target_w}:{target_h}:(in_w-{target_w})/2:0[vcomposed]"
            )
            current_video_label = "[vcomposed]"

        sub_file = None
        if burn_subtitles and transcript_segments:
            unique_sub_id = uuid.uuid4().hex[:8]

            if subtitle_style == "karaoke":
                sub_file = self.temp_dir / f"subs_{unique_sub_id}_{int(start_time)}.ass"
                self.generate_subtitles_ass(
                    transcript_segments=transcript_segments,
                    start_time=start_time,
                    end_time=end_time,
                    output_path=sub_file,
                    hook_title=hook_title,
                    karaoke=True,
                    video_path=video_path,
                    highlight_color=highlight_color,
                    font_name=font_name,
                    font_size=font_size,
                )
                escaped_sub = str(sub_file.resolve()).replace("\\", "/").replace(":", "\\:")
                filter_complex.append(
                    f"{current_video_label}ass=filename='{escaped_sub}'[vout]"
                )
                final_video_label = "[vout]"
            else:
                sub_file = self.temp_dir / f"subs_{unique_sub_id}_{int(start_time)}.srt"
                self.generate_subtitles_srt(
                    transcript_segments,
                    start_time,
                    end_time,
                    sub_file,
                    video_path=video_path,
                )
                escaped_sub = str(sub_file.resolve()).replace("\\", "/").replace(":", "\\:")
                sub_style = "FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Alignment=2,MarginV=60"
                filter_complex.append(
                    f"{current_video_label}subtitles=filename='{escaped_sub}':force_style='{sub_style}'[vout]"
                )
                final_video_label = "[vout]"
        else:
            final_video_label = current_video_label

        use_nvenc = config.video_encoder == "nvenc" or (
            config.video_encoder == "auto" and is_nvenc_available()
        )
        if use_nvenc:
            encoder_args = ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "18", "-b:v", "0", "-pix_fmt", "yuv420p"]
        else:
            encoder_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]

        temp_render_path = self.temp_dir / f"rendering_{uuid.uuid4().hex[:8]}_{output_filename}"

        seek_args = [] if is_precut else ["-ss", f"{start_time:.2f}"]

        cmd = [
            get_ffmpeg_bin(),
            "-y",
            *seek_args,
            "-i", str(video_path),
            "-t", f"{duration:.2f}",
            "-filter_complex", ";".join(filter_complex),
            "-map", final_video_label,
            "-map", "0:a",
            *encoder_args,
            "-c:a", "aac",
            "-b:a", "256k",
            "-movflags", "+faststart",
            "-shortest",
            str(temp_render_path),
        ]

        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0 and use_nvenc:
            logger.warning("NVENC encoding failed; retrying with libx264 software fallback: %s", result.stderr)
            fallback_cmd = [
                get_ffmpeg_bin(),
                "-y",
                *seek_args,
                "-i", str(video_path),
                "-t", f"{duration:.2f}",
                "-filter_complex", ";".join(filter_complex),
                "-map", final_video_label,
                "-map", "0:a",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "256k",
                "-movflags", "+faststart",
                "-shortest",
                str(temp_render_path),
            ]
            result = subprocess.run(
                fallback_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        if result.returncode != 0:
            if temp_render_path.exists():
                try:
                    temp_render_path.unlink()
                except OSError:
                    pass
            track_error(RuntimeError(f"FFmpeg render failed: {result.stderr}"), context="editor.render_clip")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr}")

        if temp_render_path.exists():
            shutil.move(str(temp_render_path), str(output_path))

        if sub_file and sub_file.exists():
            try:
                os.remove(sub_file)
            except OSError:
                pass

        return output_path

    def re_render_clip(
        self,
        clip_filename: str,
        camera_mode: str = "custom",      # "custom", "auto", "left", "center", "right", "split_screen"
        crop_x_percent: float = 0.5,      # 0.0 (far left) to 1.0 (far right)
        zoom: float = 1.0,               # 1.0 to 2.5
        filter_preset: str = "none",     # "none", "moody", "teal_orange", "vintage", "crisp", "noir"
        highlight_color: Optional[str] = None,
        font_name: Optional[str] = None,
        font_size: Optional[int] = None,
        caption_position: str = "bottom", # "bottom", "center", "top"
        trim_start_delta: float = 0.0,
        trim_end_delta: float = 0.0,
        sfx_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Re-renders an existing short clip with updated camera framing (left, center, right, custom pan),
        zoom factor (1.0x - 2.5x), color grading filters, timeline trimming, SFX, and subtitle styling in seconds.
        """
        target_clip = self.output_dir / Path(clip_filename).name
        if not target_clip.exists():
            raise FileNotFoundError(f"Clip not found: {clip_filename}")

        meta_file = target_clip.with_suffix(".json")
        meta = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
            except Exception:
                pass

        raw_start = float(meta.get("start", 0.0))
        raw_end = float(meta.get("end", 0.0))
        if raw_end <= raw_start:
            raw_end = raw_start + get_media_duration(target_clip)

        start_time = max(0.0, raw_start + float(trim_start_delta or 0.0))
        end_time = max(start_time + 1.0, raw_end + float(trim_end_delta or 0.0))
        duration = end_time - start_time
        title = meta.get("title") or target_clip.stem
        segments = meta.get("segments") or []
        source_path_str = meta.get("source_video")
        is_precut = bool(meta.get("is_precut", False))
        burn_subtitles = bool(meta.get("burn_subtitles", True))

        source_video = target_clip
        use_widescreen_source = False

        if source_path_str and Path(source_path_str).exists():
            source_video = Path(source_path_str)
            use_widescreen_source = True
        else:
            slice_cands = list(self.temp_dir.glob(f"*{target_clip.stem}*")) + list(self.temp_dir.glob("source_*.mp4"))
            for cand in slice_cands:
                if cand.exists() and cand != target_clip:
                    source_video = cand
                    use_widescreen_source = True
                    break

        orig_w, orig_h = get_video_dimensions(source_video)
        target_w, target_h = 1080, 1920
        clamped_zoom = max(1.0, min(2.5, float(zoom or 1.0)))

        COLOR_FILTERS = {
            "moody": "eq=contrast=1.15:brightness=-0.02:saturation=0.90,colorbalance=rs=-0.04:gs=-0.02:bs=0.08",
            "teal_orange": "colorbalance=rs=0.10:gs=0.02:bs=-0.10:rh=-0.08:gh=0.04:bh=0.12,eq=contrast=1.12:saturation=1.15",
            "vintage": "colorbalance=rs=0.12:gs=0.06:bs=-0.08,eq=contrast=1.05:saturation=0.92",
            "crisp": "unsharp=5:5:0.8:5:5:0.0,eq=contrast=1.20:saturation=1.10",
            "noir": "hue=s=0,eq=contrast=1.25:brightness=-0.03",
        }
        color_filter_str = COLOR_FILTERS.get((filter_preset or "none").lower(), "")

        filter_complex = []

        if use_widescreen_source:
            seek_args = [] if is_precut else ["-ss", f"{start_time:.2f}"]
            render_dur = duration

            if camera_mode == "auto":
                crop_sample_start = 0.0 if is_precut else start_time
                crop_expr = detect_dynamic_speaker_crop_expression(source_video, crop_sample_start, duration, target_w, target_h)
                chain = f"[0:v]scale=-1:{target_h}:flags=lanczos,crop={target_w}:{target_h}:x='{crop_expr}':y=0"
            elif camera_mode == "split_screen":
                half_h = target_h // 2
                chain = (
                    f"[0:v]split=2[top_in][bot_in];"
                    f"[top_in]crop=ih*9/16:ih:0:0,scale={target_w}:{half_h}:flags=lanczos[top_half];"
                    f"[bot_in]crop=ih*9/16:ih:in_w-(ih*9/16):0,scale={target_w}:{half_h}:flags=lanczos[bot_half];"
                    f"[top_half][bot_half]vstack=inputs=2"
                )
            else:
                if camera_mode == "left":
                    crop_x_percent = 0.18
                elif camera_mode == "center":
                    crop_x_percent = 0.50
                elif camera_mode == "right":
                    crop_x_percent = 0.82

                base_h = orig_h
                base_w = int(base_h * (target_w / target_h))
                crop_h = max(360, int(base_h / clamped_zoom))
                crop_w = max(202, int(crop_h * (target_w / target_h)))
                max_x = max(0, orig_w - crop_w)
                crop_x = int(max_x * max(0.0, min(1.0, float(crop_x_percent))))
                crop_y = max(0, min(orig_h - crop_h, int((orig_h - crop_h) / 2)))
                chain = f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_w}:{target_h}:flags=lanczos"
        else:
            seek_args = []
            render_dur = duration
            if clamped_zoom > 1.01:
                crop_w = int(orig_w / clamped_zoom)
                crop_h = int(orig_h / clamped_zoom)
                crop_x = int((orig_w - crop_w) * max(0.0, min(1.0, float(crop_x_percent))))
                crop_y = int((orig_h - crop_h) / 2)
                chain = f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_w}:{target_h}:flags=lanczos"
            else:
                chain = f"[0:v]scale={target_w}:{target_h}:flags=lanczos"

        if color_filter_str:
            chain += f",{color_filter_str}"

        filter_complex.append(f"{chain}[vbase]")
        current_video_label = "[vbase]"

        sub_file = None
        if burn_subtitles and segments:
            active_color = highlight_color or meta.get("highlight_color", "#facc15")
            active_font = font_name or meta.get("font_name", "Montserrat")
            active_size = font_size or meta.get("font_size", 64)
            unique_id = uuid.uuid4().hex[:6]
            sub_file = self.temp_dir / f"reedit_sub_{unique_id}.ass"

            margin_v = 280
            if caption_position == "center":
                margin_v = 850
            elif caption_position == "top":
                margin_v = 1500

            self.generate_subtitles_ass(
                transcript_segments=segments,
                start_time=0.0 if (is_precut or not use_widescreen_source) else start_time,
                end_time=duration if (is_precut or not use_widescreen_source) else end_time,
                output_path=sub_file,
                hook_title=None,
                karaoke=True,
                video_path=source_video,
                highlight_color=active_color,
                font_name=active_font,
                font_size=active_size,
                margin_v=margin_v,
            )
            escaped_sub = str(sub_file.resolve()).replace("\\", "/").replace(":", "\\:")
            filter_complex.append(f"{current_video_label}ass=filename='{escaped_sub}'[vout]")
            final_video_label = "[vout]"
        else:
            final_video_label = current_video_label

        use_nvenc = config.video_encoder == "nvenc" or (
            config.video_encoder == "auto" and is_nvenc_available()
        )
        if use_nvenc:
            encoder_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "18", "-b:v", "0", "-pix_fmt", "yuv420p"]
        else:
            encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]

        temp_render_path = self.temp_dir / f"reedit_{uuid.uuid4().hex[:6]}_{target_clip.name}"

        cmd = [
            get_ffmpeg_bin(),
            "-y",
            *seek_args,
            "-i", str(source_video),
            "-t", f"{render_dur:.2f}",
            "-filter_complex", ";".join(filter_complex),
            "-map", final_video_label,
            "-map", "0:a?",
            *encoder_args,
            "-c:a", "aac",
            "-b:a", "256k",
            "-movflags", "+faststart",
            str(temp_render_path),
        ]

        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 and use_nvenc:
            logger.warning("NVENC failed on re-render; falling back to libx264: %s", res.stderr)
            fallback_cmd = [
                get_ffmpeg_bin(),
                "-y",
                *seek_args,
                "-i", str(source_video),
                "-t", f"{render_dur:.2f}",
                "-filter_complex", ";".join(filter_complex),
                "-map", final_video_label,
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "256k",
                "-movflags", "+faststart",
                str(temp_render_path),
            ]
            res = subprocess.run(fallback_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if res.returncode != 0:
            if temp_render_path.exists():
                try:
                    temp_render_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"FFmpeg re-render failed: {res.stderr}")

        if temp_render_path.exists():
            shutil.move(str(temp_render_path), str(target_clip))

        if sub_file and sub_file.exists():
            try:
                os.remove(sub_file)
            except OSError:
                pass

        meta.update({
            "camera_mode": camera_mode,
            "crop_x_percent": crop_x_percent,
            "zoom": zoom,
            "filter_preset": filter_preset,
            "highlight_color": highlight_color or meta.get("highlight_color"),
            "caption_position": caption_position,
            "last_edited": time.time(),
        })
        try:
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2)
        except Exception:
            pass

        return {
            "name": target_clip.name,
            "title": title,
            "download_url": f"/outputs/{target_clip.name}?t={int(time.time())}",
            "meta": meta,
        }
