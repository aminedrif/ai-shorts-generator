# AI Shorts Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/downloads/)

An open-source pipeline to extract high-engagement short vertical videos from long YouTube videos or local media files. Uses Whisper for timestamped speech transcription, LLMs (Google Gemini or OpenAI) for virality scoring, and FFmpeg for 9:16 vertical re-framing and subtitle overlay.


## Features

- Fast Range Ingestion: Extracts YouTube timestamped captions in under 1 second without downloading media, detects viral moments first, and surgically downloads only the required 30-40 second video clips using HTTP byte ranges.
- YouTube and Local Media Ingestion: Process online YouTube links directly via yt-dlp or supply local MP4 files with automatic format detection.
- Timestamped Speech Recognition: Local faster-whisper model runs speech-to-text with word-level timestamps when processing local video files or videos without captions.
- AI Highlight Selection: LLM analyzes transcripts to detect punchy opening hooks, high-energy discussions, and self-contained insights.
- Face-Aware Smart Auto-Framing: Employs OpenCV facial tracking to center the active speaker automatically in 9:16 vertical mode, with blurred background and center crop options.
- Dynamic Karaoke Subtitles: Generates and burns animated word-by-word highlighted captions (ASS format) with high-contrast outlines and customizable styles.
- Hook Title Overlay Banner: Optional high-retention hook title banner in the top safe zone during the first 3.5 seconds.
- Hardware Acceleration: Fully supports NVIDIA CUDA for Whisper and NVENC (`h264_nvenc`) for ultra-fast GPU rendering.
- Glassmorphic Studio Interface: Modern SupoClip-style dark studio dashboard with a live 4-step pipeline progress tracker, 9:16 vertical cards, virality score badges, custom video player, and telemetry console.

## Architecture

1. Fast Ingestion: For YouTube URLs, fetches timestamped VTT caption tracks in seconds without video media.
2. AI Highlight Ranking: Gemini 2.5 Flash or OpenAI GPT-4o-mini identifies top viral clips and assigns virality scores (1-100).
3. Section Download: yt-dlp downloads only the selected timestamp ranges via byte-range requests.
4. Smart Auto-Framing: OpenCV detects face coordinates across frames and calculates the optimal crop offset.
5. Karaoke Video Rendering: FFmpeg cuts, scales to 9:16 vertical (1080x1920), burns animated karaoke subtitles, and exports with NVENC acceleration.

## Prerequisites

- Python 3.10 or higher
- FFmpeg installed and available on your system PATH
- An API key for Google Gemini (Google AI Studio) or OpenAI

### Installing FFmpeg

- Windows (via winget):
  winget install Gyan.FFmpeg
- macOS (via Homebrew):
  brew install ffmpeg
- Linux (Ubuntu/Debian):
  sudo apt update && sudo apt install -y ffmpeg

## Installation

1. Clone the repository:
   git clone https://github.com/aminedrif/ai-shorts-generator.git
   cd ai-shorts-generator

2. Create and activate a virtual environment:
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

3. Install project dependencies:
   pip install -r requirements.txt

4. Configure environment variables:
   Copy `.env.example` to `.env` and fill in your preferred provider credentials:
   cp .env.example .env

   Example .env configuration:
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-2.5-flash
   WHISPER_MODEL=base
   WHISPER_DEVICE=auto

## Usage

### Command Line Interface (CLI)

Generate 3 clips from a YouTube URL:
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

Process a local video file with custom options:
python main.py "path/to/video.mp4" --n 4 --ratio 9:16 --style crop_center

Available CLI options:
- `source`: YouTube URL or path to a local video file
- `--n`: Number of clips to generate (default: 3)
- `--provider`: LLM provider (`gemini` or `openai`)
- `--ratio`: Target aspect ratio (`9:16` or `1:1`)
- `--style`: Framing style (`blurred_background` or `crop_center`)
- `--resolution`: Maximum download resolution for YouTube (default: 1080)
- `--whisper-model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `--no-subtitles`: Disable burned subtitle overlays

### Web Interface

Start the local server:
python server.py

Open your browser and navigate to:
http://localhost:8000

The web interface allows you to submit URLs, configure clip options, track job progress, and preview or download the generated MP4 files directly.

## Project Structure

- `main.py`: Command-line runner
- `server.py`: FastAPI server for the web interface and background processing
- `src/config.py`: Environment configuration and directory setup
- `src/downloader.py`: Media downloader and audio extractor
- `src/transcriber.py`: Whisper speech recognition engine
- `src/analyzer.py`: LLM highlight identification engine
- `src/editor.py`: FFmpeg video cutter, scaler, and subtitle embedder
- `src/pipeline.py`: End-to-end processing pipeline orchestrator
- `web/index.html`: Web dashboard interface

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
