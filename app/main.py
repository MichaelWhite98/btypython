from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .cutout import candidate_to_dict, generate_cutout_candidates


ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
ORIGINALS_DIR = STORAGE / "originals"
TASKS_DIR = STORAGE / "tasks"
OUTPUT_DIR = STORAGE / "output"
STATIC_DIR = ROOT / "static"

for directory in [ORIGINALS_DIR, TASKS_DIR, OUTPUT_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="BTY Cutout Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=STORAGE), name="files")


class UploadTokenRequest(BaseModel):
    bizType: str
    fileName: str
    contentType: str = "image/jpeg"


class CreateTaskRequest(BaseModel):
    scene: str = "food-diary-cutout"
    imageId: str = ""
    imageUrl: str


class TaskRecord(BaseModel):
    taskId: str
    status: str
    scene: str
    imageId: str = ""
    imageUrl: str
    sourceImageUrl: str
    primaryItemId: str = ""
    items: list[dict[str, Any]] = []
    failReason: str = ""
    createdAt: str
    updatedAt: str


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def _write_task(task: TaskRecord) -> None:
    _task_path(task.taskId).write_text(task.model_dump_json(indent=2), encoding="utf-8")


def _read_task(task_id: str) -> TaskRecord:
    path = _task_path(task_id)
    if not path.exists():
        raise FileNotFoundError(task_id)
    return TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _image_path_from_url(image_url: str) -> Path:
    parsed = urlparse(image_url)
    if parsed.path.startswith("/files/originals/"):
        return STORAGE / parsed.path.removeprefix("/files/")

    temp_name = f"remote-{uuid.uuid4().hex}{Path(parsed.path).suffix or '.jpg'}"
    destination = ORIGINALS_DIR / temp_name
    urlretrieve(image_url, destination)
    return destination


def _run_task(task_id: str, base_url: str) -> None:
    task = _read_task(task_id)
    try:
        image_path = _image_path_from_url(task.imageUrl)
        candidates = generate_cutout_candidates(image_path, OUTPUT_DIR, task_id)
        items = [candidate_to_dict(candidate, base_url) for candidate in candidates]
        task.status = "succeeded"
        task.items = items
        task.primaryItemId = items[0]["id"] if items else ""
        task.updatedAt = _now_iso()
        _write_task(task)
    except Exception as exc:  # pragma: no cover
        task.status = "failed"
        task.failReason = str(exc)
        task.updatedAt = _now_iso()
        _write_task(task)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug", response_class=HTMLResponse)
def debug_page() -> str:
    return (STATIC_DIR / "debug.html").read_text(encoding="utf-8")


@app.post("/api/media/upload-token")
def create_upload_token(payload: UploadTokenRequest, request: Request) -> dict[str, Any]:
    image_id = f"img_{uuid.uuid4().hex[:12]}"
    extension = Path(payload.fileName).suffix or ".jpg"
    file_name = f"{image_id}{extension}"
    file_key = f"originals/{file_name}"
    base_url = _base_url(request)
    return {
        "uploadUrl": f"{base_url}/api/uploads/local?imageId={image_id}&fileKey={file_key}",
        "fileUrl": f"{base_url}/files/{file_key}",
        "fileKey": file_key,
        "imageId": image_id,
        "fileFieldName": "file",
        "headers": {},
        "formData": {},
    }


@app.post("/api/uploads/local")
def upload_local_file(
    imageId: str,
    fileKey: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    destination = STORAGE / fileKey
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {
        "imageId": imageId,
        "fileKey": fileKey,
        "saved": True,
    }


@app.post("/api/cutout/tasks")
def create_cutout_task(
    payload: CreateTaskRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    task_id = f"cutout_task_{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    task = TaskRecord(
        taskId=task_id,
        status="queued",
        scene=payload.scene,
        imageId=payload.imageId,
        imageUrl=payload.imageUrl,
        sourceImageUrl=payload.imageUrl,
        createdAt=now,
        updatedAt=now,
    )
    _write_task(task)
    background_tasks.add_task(_run_task, task_id, _base_url(request))
    return {"taskId": task_id, "status": "queued"}


@app.get("/api/cutout/tasks/{task_id}")
def get_cutout_task(task_id: str) -> dict[str, Any]:
    try:
        task = _read_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    return task.model_dump()


@app.post("/api/cutout/analyze-direct")
def analyze_direct(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    image_id = f"img_{uuid.uuid4().hex[:12]}"
    extension = Path(file.filename or "image.jpg").suffix or ".jpg"
    file_name = f"{image_id}{extension}"
    file_key = f"originals/{file_name}"
    destination = STORAGE / file_key
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    base_url = _base_url(request)
    payload = CreateTaskRequest(
        imageId=image_id,
        imageUrl=f"{base_url}/files/{file_key}",
    )
    return create_cutout_task(payload, request, background_tasks)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "debug.html")
