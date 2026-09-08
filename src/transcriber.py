from pathlib import Path
from typing import Dict, Any, List, Optional
from faster_whisper import WhisperModel
from src.config import config
from src.logger import logger


class VideoTranscriber:
    """Performs speech-to-text transcription with word-level timestamps."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        self.model_size = model_size or config.whisper_model
        self.device = device or config.whisper_device
        self.compute_type = compute_type or config.whisper_compute_type
        self._model = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _execute_transcription(
        self,
        model: WhisperModel,
        audio_path: Path,
        language: Optional[str] = None,
    ):
        segments_raw, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return list(segments_raw), info

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes the audio file and extracts segment and word timestamps.
        Falls back to CPU execution if GPU transcription fails.
        """
        try:
            segments_raw, info = self._execute_transcription(
                self.model, audio_path, language=language
            )
        except Exception as exc:
            if self.device != "cpu":
                logger.warning(
                    f"GPU transcription failed ({exc}). Falling back to CPU with int8 quantization."
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                segments_raw, info = self._execute_transcription(
                    self._model, audio_path, language=language
                )
            else:
                raise

        segments: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for seg in segments_raw:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 2),
                    })

            segment_data = {
                "id": seg.id,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "words": words,
            }
            segments.append(segment_data)
            full_text_parts.append(seg.text.strip())

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration": round(info.duration, 2),
            "segments": segments,
            "full_text": " ".join(full_text_parts),
        }
