from pathlib import Path
from typing import Dict, Any, List, Optional
from faster_whisper import WhisperModel
from src.config import config
from src.logger import logger


DARIJA_ARABIC_PROMPT = (
    "كلام باللغة العربية والدارجة الجزائرية والمغاربية، كلمات بالدارجة: "
    "واش، شكون، كيفاش، بزاف، مليح، صحيت، خويا، والله، هكا، زعما، برك، صح، "
    "شفت، لقطة، معليش، درك، خلاص، gaming, valorant, clutch, gg, kill."
)


def contains_arabic_script(text: str) -> bool:
    """Detects if string contains Arabic or Perso-Arabic unicode characters."""
    return any(
        '\u0600' <= ch <= '\u06FF' or
        '\u0750' <= ch <= '\u077F' or
        '\u08A0' <= ch <= '\u08FF' or
        '\uFB50' <= ch <= '\uFDFF' or
        '\uFE70' <= ch <= '\uFEFF'
        for ch in text
    )


class VideoTranscriber:
    """Performs speech-to-text transcription with word-level timestamps and Arabic/Darija support."""

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

    def set_model(self, model_size: str):
        """Updates the Whisper model size dynamically."""
        clean_model = model_size.strip().lower()
        if clean_model and clean_model != self.model_size:
            logger.info(f"Switching Whisper model from '{self.model_size}' to '{clean_model}'")
            self.model_size = clean_model
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
        initial_prompt: Optional[str] = None,
    ):
        # Normalize language
        lang = language.strip().lower() if language else None
        if lang in ("auto", "none", "", "all"):
            lang = None

        # Build prompt: for Arabic/Darija or auto-detect, inject vocabulary prompt to prevent hallucinating European languages
        prompt = initial_prompt
        if not prompt:
            if lang == "ar":
                prompt = DARIJA_ARABIC_PROMPT
            elif lang is None:
                prompt = DARIJA_ARABIC_PROMPT

        segments_raw, info = model.transcribe(
            str(audio_path),
            language=lang,
            initial_prompt=prompt,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return list(segments_raw), info

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes the audio file and extracts segment and word timestamps.
        Falls back to CPU execution if GPU transcription fails.
        """
        try:
            segments_raw, info = self._execute_transcription(
                self.model, audio_path, language=language, initial_prompt=initial_prompt
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
                    self._model, audio_path, language=language, initial_prompt=initial_prompt
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

        full_text = " ".join(full_text_parts)
        is_ar = (info.language == "ar") or contains_arabic_script(full_text)

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration": round(info.duration, 2),
            "segments": segments,
            "full_text": full_text,
            "is_arabic": is_ar,
        }

