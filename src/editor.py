import os
import re
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.config import config, get_ffmpeg_bin, is_nvenc_available
from src.downloader import sanitize_filename
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

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            cap.release()
            return None

        sample_points = np.linspace(start_time, start_time + max(0.5, duration), 15)
        detected_boxes = []

        for sec in sample_points:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35))
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


def detect_speaker_crop_x(
    video_path: Path,
    start_time: float,
    duration: float,
    target_w: int = 1080,
    target_h: int = 1920,
) -> int:
    """
    Analyzes video frames across the clip interval using OpenCV face detection
    to find the optimal horizontal crop offset that keeps the active speaker centered.
    Returns the X pixel offset for crop, or -1 to fallback to standard center.
    """
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return -1

        orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if orig_h <= 0 or orig_w <= 0:
            cap.release()
            return -1

        scale = target_h / orig_h
        scaled_w = int(orig_w * scale)
        if scaled_w <= target_w:
            cap.release()
            return 0

        default_crop_x = (scaled_w - target_w) // 2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            cap.release()
            return default_crop_x

        sample_points = np.linspace(start_time, start_time + max(0.5, duration), 12)
        crop_candidates = []

        for sec in sample_points:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) > 0:
                largest = max(faces, key=lambda b: b[2] * b[3])
                face_cx = largest[0] + largest[2] / 2
                scaled_cx = face_cx * scale
                ideal_x = int(scaled_cx - target_w / 2)
                clamped_x = max(0, min(scaled_w - target_w, ideal_x))
                crop_candidates.append(clamped_x)

        cap.release()

        if crop_candidates:
            median_crop = int(np.median(crop_candidates))
            logger.info(f"Smart face-tracking framing: crop_x={median_crop} (center was {default_crop_x})")
            return median_crop

        return default_crop_x

    except Exception as e:
        logger.warning(f"Face tracking auto-framing fell back to center crop: {e}")
        return -1


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
    ) -> Path:
        """
        Generates an ASS subtitle file with animated karaoke active-word highlighting,
        custom highlight color, speaker turn emoji detection (👨 / 👩), and optional hook title banner in the top safe zone.
        """
        duration = end_time - start_time
        ass_highlight = hex_to_ass_color(highlight_color, default="&H0015CCFA")
        clean_font = font_name.strip() if font_name else "Montserrat"
        font_size = int(font_size) if font_size else 64

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
            f"Style: KaraokeSub,{clean_font},{font_size},&H00FFFFFF,{ass_highlight},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,280,1",
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
                crop_x = detect_speaker_crop_x(video_path, crop_sample_start, duration, target_w, target_h)
                crop_expr = f"{crop_x}:0" if crop_x >= 0 else f"(in_w-{target_w})/2:0"
                filter_complex.append(
                    f"[0:v]scale=-1:{target_h}:flags=lanczos,"
                    f"crop={target_w}:{target_h}:{crop_expr}[vcomposed]"
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
            crop_x = detect_speaker_crop_x(video_path, crop_sample_start, duration, target_w, target_h)
            if crop_x >= 0:
                crop_expr = f"{crop_x}:0"
            else:
                crop_expr = f"(in_w-{target_w})/2:0"

            filter_complex.append(
                f"[0:v]scale=-1:{target_h}:flags=lanczos,"
                f"crop={target_w}:{target_h}:{crop_expr}[vcomposed]"
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
