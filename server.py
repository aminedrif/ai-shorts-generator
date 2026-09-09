import uuid
from typing import Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import config
from src.pipeline import ShortsPipeline
from src.logger import logger, error_tracker, track_error

app = FastAPI(title="AI Shorts Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)
tasks: Dict[str, Dict[str, Any]] = {}

# Serve output directory for media downloads
app.mount("/outputs", StaticFiles(directory=str(config.output_dir)), name="outputs")


class GenerateJobRequest(BaseModel):
    source: str = Field(description="YouTube link or path to local video file")
    n_clips: int = Field(default=3, ge=1, le=10)
    ratio: str = Field(default="9:16")
    style: str = Field(default="smart_crop")
    burn_subtitles: bool = Field(default=True)
    burn_hook_title: bool = Field(default=False)
    subtitle_style: str = Field(default="karaoke")
    provider: Optional[str] = Field(default=None)


def run_pipeline_worker(task_id: str, req: GenerateJobRequest):
    tasks[task_id]["status"] = "processing"
    logger.info(f"Starting job {task_id} for source '{req.source}'")
    try:
        pipeline = ShortsPipeline(llm_provider=req.provider)
        result = pipeline.run(
            source=req.source,
            n_clips=req.n_clips,
            ratio=req.ratio,
            burn_subtitles=req.burn_subtitles,
            style=req.style,
            burn_hook_title=req.burn_hook_title,
            subtitle_style=req.subtitle_style,
            task_id=task_id,
        )
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        logger.info(f"Job {task_id} completed successfully.")
    except Exception as e:
        err_rec = track_error(e, module="server_worker", task_id=task_id, context={"source": req.source})
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["error_id"] = err_rec["id"]
        tasks[task_id]["traceback"] = err_rec["traceback"]


@app.get("/api/health")
def health_check():
    return {"status": "ok", "provider": config.llm_provider}


@app.post("/api/generate")
def create_generation_job(req: GenerateJobRequest, bg_tasks: BackgroundTasks):
    if not req.source.strip():
        raise HTTPException(status_code=400, detail="Video source URL or file path is required.")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "source": req.source,
        "status": "queued",
        "result": None,
        "error": None,
    }

    bg_tasks.add_task(run_pipeline_worker, task_id, req)
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/api/clips")
def list_clips():
    clips = []
    for f in config.output_dir.glob("*.mp4"):
        clips.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified": f.stat().st_mtime,
            "download_url": f"/outputs/{f.name}",
        })
    clips.sort(key=lambda x: x["modified"], reverse=True)
    return {"clips": clips}


@app.get("/api/errors")
def get_errors(limit: int = 50, task_id: Optional[str] = None):
    """Returns recent tracked errors."""
    return {"errors": error_tracker.get_recent(limit=limit, task_id=task_id)}


@app.delete("/api/errors")
def clear_errors():
    """Clears the in-memory error tracker history."""
    error_tracker.clear()
    return {"status": "cleared"}


@app.get("/api/logs")
def get_logs(lines: int = 100):
    """Returns the most recent application log lines."""
    log_file = config.logs_dir / "app.log"
    if not log_file.exists():
        return {"lines": []}
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            return {"lines": [line.rstrip() for line in all_lines[-lines:]]}
    except Exception as e:
        return {"error": str(e), "lines": []}


# Serve frontend static files if present
WEB_DIR = config.base_dir / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

