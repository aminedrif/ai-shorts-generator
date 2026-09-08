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
        style: str = "blurred_background",
        resolution: int = 1080,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the pipeline from ingestion to rendered clips."""
        start_time = time.time()
        logger.info(f"Pipeline started for source: {source} (task: {task_id or 'cli'})")

        try:
            logger.info(f"[1/4] Ingesting media: {source}")
            media = self.downloader.process_source(source, resolution=resolution)

            logger.info(f"[2/4] Transcribing audio with Whisper...")
            transcript = self.transcriber.transcribe(media["audio_path"])

            if not transcript.get("segments"):
                raise RuntimeError("No speech detected in the audio track.")

            logger.info(f"[3/4] Analyzing transcript with AI ({self.analyzer.provider}) for highlights...")
            highlights = self.analyzer.find_highlights(transcript["segments"], n_clips=n_clips)

            logger.info(f"[4/4] Rendering {len(highlights)} vertical short clips...")
            rendered_files: List[Dict[str, Any]] = []

            for i, h in enumerate(highlights, 1):
                logger.info(f"  Rendering clip {i}/{len(highlights)}: '{h.title}' ({h.start}s to {h.end}s)")
                out_file = self.editor.render_clip(
                    video_path=media["video_path"],
                    start_time=h.start,
                    end_time=h.end,
                    title=h.title,
                    transcript_segments=transcript["segments"],
                    ratio=ratio,
                    burn_subtitles=burn_subtitles,
                    style=style,
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

            total_elapsed = round(time.time() - start_time, 2)
            logger.info(f"Pipeline completed: {len(rendered_files)} clips rendered in {total_elapsed}s.")

            return {
                "title": media["title"],
                "duration": media.get("duration", 0),
                "clips_count": len(rendered_files),
                "elapsed_seconds": total_elapsed,
                "clips": rendered_files,
            }
        except Exception as e:
            track_error(e, module="pipeline", task_id=task_id, context={"source": source})
            raise
