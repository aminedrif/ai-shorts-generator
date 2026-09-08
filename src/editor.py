import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import config, get_ffmpeg_bin, is_nvenc_available
from src.downloader import sanitize_filename
from src.logger import logger, track_error


def format_srt_time(seconds: float) -> str:
    """Converts seconds into SRT timestamp format: HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


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
        """Generates an SRT subtitle file shifted to clip-relative timestamps."""
        srt_lines = []
        counter = 1

        for seg in transcript_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            # Check overlap with clip window
            if seg_end <= start_time or seg_start >= end_time:
                continue

            # Calculate relative offset
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

    def render_clip(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        title: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
        ratio: str = "9:16",
        burn_subtitles: bool = True,
        style: str = "blurred_background",
    ) -> Path:
        """
        Cuts, scales, crops, and burns subtitles for a single short clip.
        Styles: 'blurred_background' or 'crop_center'.
        """
        clean_title = sanitize_filename(title)
        output_filename = f"{clean_title}_{int(start_time)}_{int(end_time)}.mp4"
        output_path = self.output_dir / output_filename
        duration = end_time - start_time

        # Prepare filtergraph for 9:16 (1080x1920)
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
        else:
            # Smart center crop
            filter_complex.append(
                f"[0:v]scale=-1:{target_h},"
                f"crop={target_w}:{target_h}:(in_w-{target_w})/2:0[vcomposed]"
            )
            current_video_label = "[vcomposed]"

        srt_file = None
        if burn_subtitles and transcript_segments:
            import uuid
            unique_sub_id = uuid.uuid4().hex[:8]
            srt_file = self.temp_dir / f"subs_{unique_sub_id}_{int(start_time)}.srt"
            self.generate_subtitles_srt(transcript_segments, start_time, end_time, srt_file)

            # Windows path escaping for ffmpeg subtitle filter
            escaped_srt = str(srt_file.resolve()).replace("\\", "/").replace(":", "\\:")
            sub_style = "FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Alignment=2,MarginV=60"
            filter_complex.append(
                f"{current_video_label}subtitles=filename='{escaped_srt}':force_style='{sub_style}'[vout]"
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

        cmd = [
            get_ffmpeg_bin(),
            "-y",
            "-ss", f"{start_time:.2f}",
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
                "-ss", f"{start_time:.2f}",
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

        if srt_file and srt_file.exists():
            try:
                os.remove(srt_file)
            except OSError:
                pass

        return output_path
