import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import config


class HighlightCandidate(BaseModel):
    title: str = Field(description="Short descriptive title of the clip")
    hook: str = Field(description="The opening hook or soundbite that catches attention")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    score: int = Field(description="Virality score from 1 to 100")
    reason: str = Field(description="Explanation of why this segment is engaging")


SYSTEM_PROMPT = """You are an expert video editor specialized in creating viral short-form content for TikTok, YouTube Shorts, and Instagram Reels.
Your task is to analyze the timestamped transcript of a video and select the most compelling, high-retention segments.

Evaluation criteria:
1. Hook potential: Begins with an intriguing question, bold statement, emotional punch, or high-energy moment.
2. Self-contained: The segment makes sense on its own without requiring the full context.
3. Emotional or intellectual impact: Delivers an unexpected insight, humorous quote, or strong takeaway.
4. Target duration: Each highlight MUST strictly adhere to the requested duration limits.
5. Natural language: Output the title and opening hook in the same primary language as spoken in the video.

Return ONLY a valid JSON array of objects with the following schema:
[
  {
    "title": "Brief catchy title",
    "hook": "Opening sentence",
    "start": 12.5,
    "end": 45.0,
    "score": 92,
    "reason": "Why this moment hooks viewers"
  }
]
Do not wrap in additional commentary or conversational text.
"""


class HighlightAnalyzer:
    """Uses an LLM to identify the highest potential viral segments from video transcripts."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or config.llm_provider).lower()

    def _prepare_transcript_text(self, segments: List[Dict[str, Any]]) -> str:
        lines = []
        for s in segments:
            lines.append(f"[{s['start']:.2f}s -> {s['end']:.2f}s]: {s['text']}")
        return "\n".join(lines)

    def _clean_json_response(self, text: str) -> List[Dict[str, Any]]:
        # Strip markdown fences if present
        cleaned = re.sub(r"^```json\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()

        # Find array boundaries
        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx : end_idx + 1]

        data = json.loads(cleaned)
        if isinstance(data, dict) and "highlights" in data:
            data = data["highlights"]
        return data

    def _analyze_with_gemini(
        self,
        transcript_str: str,
        n_clips: int,
        heatmap_context: str = "",
        min_duration: int = 20,
        max_duration: int = 60,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
    ) -> List[HighlightCandidate]:
        import google.generativeai as genai

        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env")

        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=config.gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )

        window_instruction = ""
        if time_range_start is not None or time_range_end is not None:
            w_start = time_range_start or 0.0
            w_end = f"{time_range_end:.1f}s" if time_range_end else "the specified window end"
            window_instruction = f"MANDATORY TIME WINDOW: All selected clips must fall strictly between {w_start:.1f}s and {w_end}.\n"

        prompt = (
            f"{heatmap_context}{window_instruction}"
            f"Analyze the following timestamped transcript and return the top {n_clips} best viral clips. "
            f"Each clip MUST be strictly between {min_duration} and {max_duration} seconds in duration.\n\n"
            f"{transcript_str}"
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3},
        )

        raw_json = self._clean_json_response(response.text)
        return [HighlightCandidate(**item) for item in raw_json][:n_clips]

    def _analyze_with_openai(
        self,
        transcript_str: str,
        n_clips: int,
        heatmap_context: str = "",
        min_duration: int = 20,
        max_duration: int = 60,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
    ) -> List[HighlightCandidate]:
        from openai import OpenAI

        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        client = OpenAI(api_key=config.openai_api_key)
        window_instruction = ""
        if time_range_start is not None or time_range_end is not None:
            w_start = time_range_start or 0.0
            w_end = f"{time_range_end:.1f}s" if time_range_end else "the specified window end"
            window_instruction = f"MANDATORY TIME WINDOW: All selected clips must fall strictly between {w_start:.1f}s and {w_end}.\n"

        prompt = (
            f"{heatmap_context}{window_instruction}"
            f"Analyze the following timestamped transcript and return the top {n_clips} best viral clips. "
            f"Each clip MUST be strictly between {min_duration} and {max_duration} seconds in duration.\n\n"
            f"{transcript_str}"
        )

        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content or "[]"
        raw_json = self._clean_json_response(content)
        return [HighlightCandidate(**item) for item in raw_json][:n_clips]

    def find_highlights(
        self,
        transcript_segments: List[Dict[str, Any]],
        n_clips: int = 3,
        heatmap_peaks: Optional[List[Dict[str, Any]]] = None,
        min_duration: int = 20,
        max_duration: int = 60,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
    ) -> List[HighlightCandidate]:
        """Detects top highlight moments based on transcript, duration bounds, and audience retention peaks."""
        transcript_str = self._prepare_transcript_text(transcript_segments)

        heatmap_context = ""
        if heatmap_peaks:
            lines = ["\nREAL AUDIENCE RETENTION DATA (YouTube 'Most Replayed' Heatmap Peaks):"]
            for i, p in enumerate(heatmap_peaks[:5], 1):
                lines.append(
                    f"- Peak #{i}: {p['start_time']}s to {p['end_time']}s (Audience Replay Intensity: {int(p['intensity']*100)}%)"
                )
            lines.append("PRIORITIZE selecting clips around these proven high-retention windows where viewers replayed the video the most.")
            heatmap_context = "\n".join(lines) + "\n\n"

        candidates: List[HighlightCandidate] = []
        try:
            if self.provider == "gemini":
                candidates = self._analyze_with_gemini(
                    transcript_str,
                    n_clips,
                    heatmap_context=heatmap_context,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    time_range_start=time_range_start,
                    time_range_end=time_range_end,
                )
            elif self.provider == "openai":
                candidates = self._analyze_with_openai(
                    transcript_str,
                    n_clips,
                    heatmap_context=heatmap_context,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    time_range_start=time_range_start,
                    time_range_end=time_range_end,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'gemini' or 'openai'.")
        except Exception:
            candidates = []

        # Fallback using YouTube heatmap peaks directly if available
        if not candidates and heatmap_peaks and transcript_segments:
            for i, p in enumerate(heatmap_peaks[:n_clips], 1):
                pk_s = p["start_time"]
                matching_seg = next(
                    (s for s in transcript_segments if abs(s["start"] - pk_s) < 20.0),
                    transcript_segments[0],
                )
                start_c = max(0.0, pk_s - 2.0)
                end_c = min(float(transcript_segments[-1].get("end", start_c + 35.0)), start_c + 38.0)
                hook_txt = matching_seg.get("text", "Most Replayed Moment")[:60].strip()
                intensity_pct = int(p["intensity"] * 100)
                candidates.append(
                    HighlightCandidate(
                        title=f"Viral Peak #{i}",
                        hook=hook_txt,
                        start=round(start_c, 2),
                        end=round(end_c, 2),
                        score=min(99, int(88 + p["intensity"] * 11)),
                        reason=f"Aligned with YouTube's #{i} Most Replayed spike ({intensity_pct}% viewer retention).",
                    )
                )

        # Generic fallback if no highlights returned
        if not candidates and transcript_segments:
            start_t = float(transcript_segments[0].get("start", 0.0))
            end_t = float(transcript_segments[-1].get("end", start_t + 15.0))
            first_text = transcript_segments[0].get("text", "Key Highlight")
            candidates.append(
                HighlightCandidate(
                    title="Key Highlight",
                    hook=first_text[:60].strip(),
                    start=start_t,
                    end=min(end_t, round(start_t + max(min_duration, min(max_duration, 45.0)), 2)),
                    score=85,
                    reason="Automatically generated highlight covering key transcript moments.",
                )
            )

        # Sort by virality score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
