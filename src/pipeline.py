import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.downloader import VideoDownloader
from src.transcriber import VideoTranscriber
from src.analyzer import HighlightAnalyzer
from src.editor import VideoEditor
from src.logger import logger, track_error


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
    ) -> Dict[str, Any]:
        """Runs the pipeline from ingestion to rendered clips."""
        start_time = time.time()
        logger.info(f"Pipeline started for source: {source} (task: {task_id or 'cli'})")

        try:
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

            if not fast_ingestion:
                logger.info(f"[1/4] Ingesting media: {source}")
                media = self.downloader.process_source(source, resolution=resolution)
                media_title = media["title"]
                media_duration = media.get("duration", 0)

                logger.info(f"[2/4] Transcribing audio with Whisper...")
                transcript = self.transcriber.transcribe(media["audio_path"])

            if not transcript or not transcript.get("segments"):
                raise RuntimeError("No speech or subtitles detected for this video.")

            logger.info(
                f"[3/4] Analyzing transcript with AI ({self.analyzer.provider}) for {n_clips} viral moments..."
            )
            highlights = self.analyzer.find_highlights(transcript["segments"], n_clips=n_clips)

            logger.info(f"[4/4] Rendering {len(highlights)} vertical short clips...")
            rendered_files: List[Dict[str, Any]] = []

            for i, h in enumerate(highlights, 1):
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
