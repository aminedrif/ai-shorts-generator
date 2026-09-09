# AI Shorts Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-7.0+-007808.svg)](https://ffmpeg.org)

An open-source AI pipeline and dark studio dashboard to extract high-engagement, vertical short clips (9:16) from any web link containing a video, social media, YouTube, or local media files. 

The pipeline combines Whisper speech transcription, YouTube audience retention heatmaps, LLM virality scoring (Google Gemini or OpenAI), OpenCV facial tracking, and FFmpeg rendering with animated karaoke subtitles.

---

## Key Features

- **Universal Video Link Ingestion**: Accepts links from any website across the internet:
  - Any web page or blog with embedded HTML5 video or OpenGraph metadata (`og:video`, `<video>`, JSON-LD VideoObject).
  - Video sharing and social platforms: YouTube, TikTok, Instagram Reels, X / Twitter, Vimeo, Reddit, Twitch, Facebook, Loom, and DailyMotion.
  - Direct video files and HLS/DASH streams: `.mp4`, `.webm`, `.mov`, `.mkv`, `.m4v`, `.m3u8`, and `.mpd`.
  - Local video file uploads (MP4, MOV, AVI) with automatic media inspection.
  - Multi-tier fallback architecture: yt-dlp with desktop browser headers, deep HTML video scraping, and direct FFmpeg stream copy.
- **YouTube Audience Retention Heatmap Alignment**: Automatically extracts YouTube's "Most Replayed" viewer retention curve data, aligning AI highlight candidates with real audience attention peaks.
- **Sub-Frame Subtitle Synchronization**: Parses YouTube `json3` per-word millisecond offsets (`tOffsetMs`) and runs local Whisper word-level timestamps to ensure sub-frame subtitle alignment, even during rapid speech.
- **Fast Range Ingestion**: For YouTube videos with captions, retrieves transcript cues in 1-2 seconds without downloading video media, detects viral moments first, and surgically downloads only the selected 30-60 second ranges via HTTP byte requests.
- **AI Virality & Highlight Ranking**: Analyzes transcript segments using Google Gemini 2.5 Flash or OpenAI GPT-4o-mini to score opening hooks, discussion energy, and standalone shareability.
- **Face-Aware Smart Auto-Framing**: Employs OpenCV facial detection to continuously track active speakers and calculate optimal crop offsets for vertical 9:16 framing. Supports smart crop, speaker pan, split-screen, and blurred background framing.
- **Dynamic Karaoke Subtitles**: Burns animated word-by-word highlighted captions (ASS format) with customizable font styles (Karaoke Highlighter, Bold Impact, Minimal Clean), colors, and font sizes.
- **Interactive iPhone 16 Studio Mockup**: Live 9:16 mobile player chassis inside the dashboard with real-time video playback for YouTube embeds, direct video streams, uploaded files, and rendered clips, with audio toggles and play/pause controls.
- **Task Cancellation & Refresh Persistence**: Real-time generation cancellation controls (`POST /api/cancel/{task_id}`) to halt processing immediately, combined with browser localStorage and server-side task caching (`/api/tasks/active`) to recover in-flight jobs across page reloads.
- **4-Metric Virality Breakdown & Clip Management**: Detailed scores for Hook, Engagement, Value, and Shareability (out of 100), transcript quote box, direct MP4 downloads, and safe clip deletion with a confirmation modal (`DELETE /api/clips/{filename}`).
- **Hardware Acceleration**: Supports NVIDIA CUDA for Whisper transcription and NVENC (`h264_nvenc`) for accelerated FFmpeg rendering.

---

## Architecture Overview

```
Any Video Link / File
   │
   ├── [YouTube with Captions] ──> Fast Ingestion (json3/VTT cues + Replay Heatmap) ──┐
   │                                                                                  │
   └── [Any Website / Stream]  ──> Universal Downloader + Whisper Transcription ─────┤
                                                                                      │
                                                                                      ▼
                                                                           AI Virality Analyzer
                                                                      (Gemini 2.5 Flash / GPT-4o-mini)
                                                                                      │
                                                                                      ▼
                                                                           Highlight Selection
                                                                       (Hook, Virality Score, Ranges)
                                                                                      │
                                                                                      ▼
                                                                          OpenCV Face Auto-Framing
                                                                           (Active Speaker Tracking)
                                                                                      │
                                                                                      ▼
                                                                        FFmpeg 9:16 Video Rendering
                                                                       (Karaoke Captions + Hook Banner)
                                                                                      │
                                                                                      ▼
                                                                        Studio Gallery & iPhone Mockup
```

---

## Prerequisites

- **Python 3.10** or higher
- **FFmpeg 7.0+** installed and available on your system PATH
- An API key for **Google Gemini** (Google AI Studio) or **OpenAI**

### Installing FFmpeg

- **Windows** (via winget):
  ```powershell
  winget install Gyan.FFmpeg
  ```
- **macOS** (via Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Linux** (Ubuntu / Debian):
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aminedrif/ai-shorts-generator.git
   cd ai-shorts-generator
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory (or copy from `.env.example`):
   ```ini
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   WHISPER_MODEL=base
   WHISPER_DEVICE=auto
   OUTPUT_DIR=outputs
   TEMP_DIR=temp
   ```

---

## Usage

### Web Studio Interface

Start the local server:
```bash
python server.py
```
Or via Uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
```
http://localhost:8000
```

The web studio features:
- **Video Link Tab**: Paste any link (YouTube, TikTok, Instagram, Twitter, any website, or direct stream).
- **Upload Video Tab**: Drag and drop local MP4, MOV, or AVI files.
- **iPhone 16 Mockup**: Watch your video run live inside the 9:16 mobile chassis with sound and play/pause toggles.
- **Live Progress Stepper**: 6-step progress bar with active task cancellation.
- **Clip Gallery**: View 4-metric virality breakdown, preview clips in the mockup, download MP4s, or delete clips with confirmation.

### Command Line Interface (CLI)

Generate 3 clips from a YouTube URL:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Process any website or direct video link:
```bash
python main.py "https://example.com/stream.mp4" --n 3 --ratio 9:16
```

Process a local video file:
```bash
python main.py "path/to/video.mp4" --n 4 --ratio 9:16 --style smart_crop
```

#### CLI Options

| Option | Description | Default |
|---|---|---|
| `source` | Any video URL or local file path | Required |
| `--n` | Number of clips to generate | `3` |
| `--provider` | LLM provider (`gemini` or `openai`) | `gemini` |
| `--ratio` | Aspect ratio (`9:16` or `1:1`) | `9:16` |
| `--style` | Framing style (`smart_crop`, `speaker_pan`, `split_screen`, `original`) | `smart_crop` |
| `--resolution` | Video download resolution height | `1080` |
| `--whisper-model` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`) | `base` |
| `--subtitle-style` | Subtitle style (`karaoke`, `impact`, `clean`, `default`) | `karaoke` |
| `--no-subtitles` | Disable burned subtitle overlays | `False` |
| `--burn-hook` | Burn hook title banner in first 3.5 seconds | `False` |

---

## Project Structure

```
ai-shorts-generator/
├── main.py              # CLI entrypoint
├── server.py            # FastAPI server with background tasks and preview API
├── requirements.txt     # Python package dependencies
├── .env.example         # Environment template
├── src/
│   ├── analyzer.py      # AI highlight detection and virality scoring
│   ├── config.py        # Settings and environment resolution
│   ├── downloader.py    # Universal downloader, deep HTML scraper, and range cutter
│   ├── editor.py        # FFmpeg vertical formatter, face tracker, and karaoke subtitle burner
│   ├── logger.py        # Logging system and error telemetry tracker
│   ├── pipeline.py      # End-to-end orchestration pipeline
│   └── transcriber.py   # faster-whisper speech recognition engine
├── web/
│   └── index.html       # Studio web application and iPhone 16 live mockup
├── outputs/             # Rendered MP4 clips and JSON metadata
└── temp/                # Working directory for audio, slices, and subtitles
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
