import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from src.downloader import VideoDownloader
from src.transcriber import VideoTranscriber
from src.analyzer import HighlightAnalyzer
from src.editor import VideoEditor
from src.logger import logger, track_error

# Registry of task IDs that have been requested to cancel
CANCELLED_TASKS = set()


class ShortsPipeline:
    """Orchestrates the entire short video generation process."""

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        whisper_model: Optional[str] = None,
    ):
        self.downloader = VideoDownloader()
        self.transcriber = VideoTranscriber(model_size=whisper_model)
        self.analyzer = HighlightAnalyzer(provider=llm_provider)
        self.editor = VideoEditor()

    def run(
        self,
        source: str,
        n_clips: int = 3,
        ratio: str = "9:16",
        burn_subtitles: bool = True,
        style: str = "smart_crop",
        resolution: int = 1080,
        task_id: Optional[str] = None,
        burn_hook_title: bool = False,
        subtitle_style: str = "karaoke",
        progress_callback: Optional[Callable[[int, str, str], None]] = None,
        language: Optional[str] = None,
        whisper_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the pipeline from ingestion to rendered clips."""
        start_time = time.time()
        logger.info(f"Pipeline started for source: {source} (task: {task_id or 'cli'})")

        def report(pct: int, step_name: str, message: str):
            if task_id and task_id in CANCELLED_TASKS:
                logger.info(f"Task {task_id} cancellation detected during step '{step_name}'. Aborting.")
                raise RuntimeError("Task was cancelled by user.")
            if progress_callback:
                try:
                    progress_callback(pct, step_name, message)
                except Exception as pe:
                    logger.debug(f"Progress callback error: {pe}")

        try:
            report(10, "ingestion", "Fetching video metadata and caption cues...")

            is_youtube = source.startswith(("http://", "https://")) and (
                "youtube.com" in source or "youtu.be" in source
            )
            transcript: Optional[Dict[str, Any]] = None
            media: Optional[Dict[str, Any]] = None
            fast_ingestion = False

            if is_youtube:
                logger.info(f"[1/4] Fast Ingestion: Extracting transcript cues for YouTube URL...")
                transcript = self.downloader.fetch_youtube_transcript(source)
                if transcript and transcript.get("segments"):
                    fast_ingestion = True
                    media_title = transcript.get("title", "YouTube Video")
                    media_duration = transcript.get("duration", 0)
                    logger.info(
                        f"[2/4] Direct transcript acquired ({len(transcript['segments'])} cues). Local Whisper transcription skipped."
                    )
                    report(30, "transcript", f"Retrieved {len(transcript['segments'])} captions and audience retention heatmap.")

            if not fast_ingestion:
                logger.info(f"[1/4] Ingesting media: {source}")
                report(20, "ingestion", "Downloading video source...")
                media = self.downloader.process_source(source, resolution=resolution)
                media_title = media["title"]
                media_duration = media.get("duration", 0)

                if whisper_model:
                    self.transcriber.set_model(whisper_model)
                logger.info(f"[2/4] Transcribing audio with Whisper (model: {self.transcriber.model_size}, lang: {language or 'auto'})...")
                report(40, "transcript", f"Transcribing audio with Whisper ({self.transcriber.model_size}, {language or 'auto-detect'})...")
                transcript = self.transcriber.transcribe(media["audio_path"], language=language)

            if not transcript or not transcript.get("segments"):
                raise RuntimeError("No speech or subtitles detected for this video.")

            heatmap_peaks = transcript.get("heatmap_peaks")
            if heatmap_peaks:
                logger.info(
                    f"YouTube Heatmap: Aligning with {len(heatmap_peaks)} Most Replayed peaks (Top peak: {int(heatmap_peaks[0]['intensity']*100)}% retention)."
                )

            report(50, "analysis", "AI analyzing retention peaks and virality scoring...")
            highlights = self.analyzer.find_highlights(
                transcript["segments"], n_clips=n_clips, heatmap_peaks=heatmap_peaks
            )

            logger.info(f"[4/4] Rendering {len(highlights)} vertical short clips...")
            rendered_files: List[Dict[str, Any]] = []

            for i, h in enumerate(highlights, 1):
                clip_pct = 55 + int(((i - 1) / len(highlights)) * 40)
                report(clip_pct, "rendering", f"Rendering clip {i}/{len(highlights)}: '{h.title}'...")
                logger.info(f"  Processing clip {i}/{len(highlights)}: '{h.title}' ({h.start}s to {h.end}s)")
                temp_slice: Optional[Path] = None

                try:
                    if fast_ingestion:
                        slice_id = f"{task_id or 'job'}_{i}_{int(h.start)}"
                        temp_slice = self.downloader.temp_dir / f"slice_{slice_id}.mp4"
                        logger.info(f"  Surgically downloading range {h.start}s - {h.end}s via HTTP range requests...")
                        input_video = self.downloader.download_video_section(
                            url=source,
                            start_time=h.start,
                            end_time=h.end,
                            output_path=temp_slice,
                            resolution=resolution,
                        )
                        is_precut = True
                    else:
                        input_video = media["video_path"]
                        is_precut = False

                    out_file = self.editor.render_clip(
                        video_path=input_video,
                        start_time=h.start,
                        end_time=h.end,
                        title=h.title,
                        transcript_segments=transcript["segments"],
                        ratio=ratio,
                        burn_subtitles=burn_subtitles,
                        style=style,
                        hook_title=(h.hook or h.title) if burn_hook_title else None,
                        subtitle_style=subtitle_style,
                        is_precut=is_precut,
                    )

                    meta_data = {
                        "title": h.title,
                        "hook": h.hook,
                        "score": h.score,
                        "start": h.start,
                        "end": h.end,
                        "reason": h.reason,
                    }
                    meta_path = out_file.with_suffix(".json")
                    try:
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump(meta_data, mf, indent=2)
                    except Exception as me:
                        logger.warning(f"Could not save clip metadata: {me}")

                    rendered_files.append({
                        "title": h.title,
                        "hook": h.hook,
                        "score": h.score,
                        "start": h.start,
                        "end": h.end,
                        "reason": h.reason,
                        "file_path": str(out_file),
                        "file_name": out_file.name,
                    })

                finally:
                    if temp_slice and temp_slice.exists():
                        try:
                            temp_slice.unlink()
                        except OSError:
                            pass

            report(100, "complete", f"Completed! Rendered {len(rendered_files)} clips.")
            total_elapsed = round(time.time() - start_time, 2)
            logger.info(f"Pipeline completed: {len(rendered_files)} clips rendered in {total_elapsed}s.")

            return {
                "title": media_title,
                "duration": media_duration,
                "clips_count": len(rendered_files),
                "elapsed_seconds": total_elapsed,
                "clips": rendered_files,
                "fast_ingestion": fast_ingestion,
            }
        except Exception as e:
            track_error(e, module="pipeline", task_id=task_id, context={"source": source})
            raise
