import os
import uuid
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import config, get_ffmpeg_bin, is_nvenc_available
from src.downloader import sanitize_filename
from src.logger import logger, track_error


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
    ) -> Path:
        """Generates a standard SRT subtitle file shifted to clip-relative timestamps."""
        srt_lines = []
        counter = 1

        for seg in transcript_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            if seg_end <= start_time or seg_start >= end_time:
                continue

            rel_start = max(0.0, seg_start - start_time)
            rel_end = min(end_time - start_time, seg_end - start_time)

            if rel_end - rel_start < 0.1:
                continue

            text = seg["text"].strip()
            if not text:
                continue

            srt_lines.append(str(counter))
            srt_lines.append(f"{format_srt_time(rel_start)} --> {format_srt_time(rel_end)}")
            srt_lines.append(text)
            srt_lines.append("")
            counter += 1

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
    ) -> Path:
        """
        Generates an ASS subtitle file with animated karaoke active-word highlighting
        and optional hook title banner in the top safe zone.
        """
        duration = end_time - start_time
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
            "Style: KaraokeSub,Arial,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,280,1",
            "Style: HookTitle,Arial,48,&H00FFFFFF,&H00000000,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,3,12,0,8,60,60,180,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        events = []

        # 1. Hook title banner in top safe area for first ~3.5 seconds
        if hook_title:
            banner_end = min(3.5, max(1.5, duration))
            clean_hook = hook_title.replace("\n", " ").strip().upper()
            if clean_hook:
                events.append(
                    f"Dialogue: 1,0:00:00.20,{format_ass_time(banner_end)},HookTitle,,0,0,0,,{{\\fad(200,400)}}{clean_hook}"
                )

        # 2. Extract words falling inside [start_time, end_time]
        valid_words = []
        for seg in transcript_segments:
            for w in seg.get("words", []):
                w_start = float(w["start"])
                w_end = float(w["end"])
                if w_end > start_time and w_start < end_time:
                    rel_s = max(0.0, w_start - start_time)
                    rel_e = min(duration, w_end - start_time)
                    if rel_e > rel_s:
                        valid_words.append({
                            "word": w["word"].strip().upper(),
                            "start": rel_s,
                            "end": rel_e,
                        })

        if karaoke and valid_words:
            # Group words into natural chunks of 3-4 words or punctuation breaks
            chunks = []
            current_chunk = []
            for w in valid_words:
                current_chunk.append(w)
                if len(current_chunk) >= 4 or any(w["word"].endswith(p) for p in [".", "!", "?", ","]):
                    chunks.append(current_chunk)
                    current_chunk = []
            if current_chunk:
                chunks.append(current_chunk)

            for chunk in chunks:
                for i, active_w in enumerate(chunk):
                    w_start = active_w["start"]
                    if i < len(chunk) - 1:
                        w_end = min(active_w["end"], chunk[i + 1]["start"])
                    else:
                        w_end = active_w["end"]

                    if w_end <= w_start:
                        w_end = w_start + 0.1

                    formatted_parts = []
                    for j, w in enumerate(chunk):
                        if j < i:
                            formatted_parts.append(f"{{\\c&H00C0C0C0}}{w['word']}")
                        elif j == i:
                            formatted_parts.append(f"{{\\c&H0000FFFF\\t(0,80,\\fscx108\\fscy108)}}{w['word']}")
                        else:
                            formatted_parts.append(f"{{\\c&H00FFFFFF}}{w['word']}")

                    line_text = " ".join(formatted_parts)
                    events.append(
                        f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},KaraokeSub,,0,0,0,,{line_text}"
                    )
        else:
            # Fallback to segment-level subtitles
            for seg in transcript_segments:
                seg_s = float(seg["start"])
                seg_e = float(seg["end"])
                if seg_e > start_time and seg_s < end_time:
                    rel_s = max(0.0, seg_s - start_time)
                    rel_e = min(duration, seg_e - start_time)
                    text = seg["text"].strip().upper()
                    if text:
                        events.append(
                            f"Dialogue: 0,{format_ass_time(rel_s)},{format_ass_time(rel_e)},KaraokeSub,,0,0,0,,{{\\c&H00FFFFFF}}{text}"
                        )

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

        if style == "blurred_background":
            filter_complex.append(
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h},boxblur=20:5[bgblurred];"
                f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fgscaled];"
                f"[bgblurred][fgscaled]overlay=(W-w)/2:(H-h)/2[vcomposed]"
            )
            current_video_label = "[vcomposed]"
        elif style in ("smart_crop", "crop_smart"):
            crop_sample_start = 0.0 if is_precut else start_time
            crop_x = detect_speaker_crop_x(video_path, crop_sample_start, duration, target_w, target_h)
            if crop_x >= 0:
                crop_expr = f"{crop_x}:0"
            else:
                crop_expr = f"(in_w-{target_w})/2:0"

            filter_complex.append(
                f"[0:v]scale=-1:{target_h},"
                f"crop={target_w}:{target_h}:{crop_expr}[vcomposed]"
            )
            current_video_label = "[vcomposed]"
        else:
            # Standard center crop
            filter_complex.append(
                f"[0:v]scale=-1:{target_h},"
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
                    hook_title=hook_title or title,
                    karaoke=True,
                )
                escaped_sub = str(sub_file.resolve()).replace("\\", "/").replace(":", "\\:")
                filter_complex.append(
                    f"{current_video_label}ass=filename='{escaped_sub}'[vout]"
                )
                final_video_label = "[vout]"
            else:
                sub_file = self.temp_dir / f"subs_{unique_sub_id}_{int(start_time)}.srt"
                self.generate_subtitles_srt(transcript_segments, start_time, end_time, sub_file)
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
            encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22", "-pix_fmt", "yuv420p"]
        else:
            encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p"]

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
            "-b:a", "192k",
            str(output_path),
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
                "-preset", "fast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                str(output_path),
            ]
            result = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            track_error(RuntimeError(f"FFmpeg render failed: {result.stderr}"), context="editor.render_clip")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr}")

        if sub_file and sub_file.exists():
            try:
                os.remove(sub_file)
            except OSError:
                pass

        return output_path
