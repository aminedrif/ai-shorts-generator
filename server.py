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
    style: str = Field(default="blurred_background")
    burn_subtitles: bool = Field(default=True)
    provider: Optional[str] = Field(default=None)


def run_pipeline_worker(task_id: str, req: GenerateJobRequest):
    tasks[task_id]["status"] = "processing"
    try:
        pipeline = ShortsPipeline(llm_provider=req.provider)
        result = pipeline.run(
            source=req.source,
            n_clips=req.n_clips,
            ratio=req.ratio,
            burn_subtitles=req.burn_subtitles,
            style=req.style,
        )
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


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


# Serve frontend static files if present
WEB_DIR = config.base_dir / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

