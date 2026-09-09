import argparse
import sys
from src.pipeline import ShortsPipeline
from src.config import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Shorts Generator: Turn long videos into engaging vertical short clips."
    )
    parser.add_argument(
        "source",
        type=str,
        help="YouTube video URL or path to a local video file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of short clips to generate (default: 3).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openai"],
        default=config.llm_provider,
        help="LLM provider for highlight detection (default from .env).",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        choices=["9:16", "1:1"],
        default="9:16",
        help="Aspect ratio for output videos (default: 9:16).",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["blurred_background", "crop_center"],
        default="blurred_background",
        help="Framing style for 9:16 conversion (default: blurred_background).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1080,
        help="Target download resolution for YouTube videos (default: 1080).",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Disable burning subtitles onto the output clips.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=config.whisper_model,
        help="Whisper model size (tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Audio language (e.g., 'ar' for Arabic / Algerian Darija, 'en', 'fr', or 'auto').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = ShortsPipeline(
        llm_provider=args.provider,
        whisper_model=args.whisper_model,
    )

    try:
        result = pipeline.run(
            source=args.source,
            n_clips=args.n,
            ratio=args.ratio,
            burn_subtitles=not args.no_subtitles,
            style=args.style,
            resolution=args.resolution,
            language=args.language,
            whisper_model=args.whisper_model,
        )

        print("\nSummary:")
        print(f"Title: {result['title']}")
        print(f"Generated clips ({len(result['clips'])}):")
        for clip in result["clips"]:
            print(f"- [{clip['score']}/100] {clip['title']} -> {clip['file_path']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
