# AI Shorts Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/downloads/)

An open-source pipeline to extract high-engagement short vertical videos from long YouTube videos or local media files. Uses Whisper for timestamped speech transcription, LLMs (Google Gemini or OpenAI) for virality scoring, and FFmpeg for 9:16 vertical re-framing and subtitle overlay.


## Features

- YouTube and Local Media Ingestion: Download YouTube links directly via yt-dlp or supply local MP4 files.
- Timestamped Speech Recognition: Local faster-whisper model runs speech-to-text with word-level timestamps.
- AI Highlight Selection: LLM analyzes transcripts to detect punchy opening hooks, high-energy discussions, and self-contained insights.
- 9:16 Vertical Video Re-framing: Formats 16:9 landscape content into vertical shorts using blurred background padding or smart center cropping.
- Subtitle Burning: Generates and embeds synchronized, readable subtitles directly into the output video.
- Dual Interface: Use the CLI for automation and scripts, or run the web UI for interactive job queueing and video playback.

## Architecture

1. Ingestion: yt-dlp downloads the source video and extracts a 16kHz mono WAV track.
2. Transcription: faster-whisper outputs timestamped segments and words.
3. Analysis: The chosen LLM (Gemini 2.5 Flash or OpenAI GPT-4o-mini) ranks candidates from 1 to 100 based on hook strength, emotional impact, and conciseness.
4. Video Rendering: FFmpeg cuts the selected time window, scales to 9:16 vertical (1080x1920), and applies styled subtitle captions.
5. Storage: Output videos are saved to the output directory.

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
