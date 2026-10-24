"""Local API for the PV cell inspection console.

This wrapper deliberately keeps model loading lazy. The existing training,
weights, dataset, and inference scripts remain the source of truth; this file
only exposes them through HTTP for the frontend.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "inspection_test" / "api_uploads"
OUTPUTS = BASE / "inspection_test" / "api_outputs"
DATA_YAML = BASE / "yolo_dataset_repaired_8class" / "data.yaml"
MODEL_YAML = BASE / "models" / "yolo26_ghostlite.yaml"
WEIGHTS = BASE / "PVEL_Detection_Repaired" / "ghostlite" / "weights" / "best.pt"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b-instruct")

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PVEL-AD Inspection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_detector: Any = None
_detector_lock = threading.Lock()
_history: list[dict[str, Any]] = []


def class_names() -> dict[int, str]:
    with DATA_YAML.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)["names"]
    if isinstance(raw, list):
        return {index: name for index, name in enumerate(raw)}
    return {int(index): name for index, name in raw.items()}


def detector_status() -> dict[str, Any]:
    dependencies: dict[str, bool] = {}
    for module in ("torch", "ultralytics"):
        try:
            __import__(module)
            dependencies[module] = True
        except Exception:
            dependencies[module] = False
    configured = MODEL_YAML.is_file() and WEIGHTS.is_file() and DATA_YAML.is_file()
    return {
        "configured": configured,
        "dependencies": dependencies,
        "model": "YOLO26-GhostLite",
        "weights": str(WEIGHTS),
    }


def get_detector() -> Any:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                from custom_yolo import load_custom_yolo

                model = load_custom_yolo(MODEL_YAML)
                model.load(str(WEIGHTS))
                _detector = model
    return _detector


def ollama_status() -> dict[str, Any]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        response.raise_for_status()
        models = response.json().get("models", [])
        names = {item.get("name") for item in models}
        return {"connected": True, "model": OLLAMA_MODEL, "model_available": OLLAMA_MODEL in names}
    except (requests.RequestException, ValueError):
        return {"connected": False, "model": OLLAMA_MODEL, "model_available": False}


def cautious_prompt(detections: list[dict[str, Any]]) -> str:
    return f"""You are assisting with preliminary solar-cell electroluminescence inspection.
Only discuss the detector predictions below. Do not invent findings.
Predictions: {json.dumps(detections, indent=2)}
Explain the observed classes, confidence, and cautious follow-up actions.
Clearly distinguish model predictions from verified physical defects.
Do not claim a panel is safe or unsafe, or that repair is definitely required,
based on one image. If evidence is insufficient, say so."""


def ask_ollama(image_bytes: bytes, detections: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    status = ollama_status()
    if not status["connected"] or not status["model_available"]:
        return None, status
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{
                    "role": "user",
                    "content": cautious_prompt(detections),
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }],
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"], status
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        return f"LLM explanation unavailable: {exc}", {**status, "connected": False}


def run_detector(path: Path, output_path: Path) -> list[dict[str, Any]]:
    model = get_detector()
    names = class_names()
    device = "0" if __import__("torch").cuda.is_available() else "cpu"
    results = model.predict(source=str(path), imgsz=640, conf=0.25, device=device, verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        plotted = result.plot()
        from PIL import Image

        Image.fromarray(plotted[:, :, ::-1]).save(output_path)
        image_height, image_width = result.orig_shape
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "class_name": names.get(class_id, f"unknown_{class_id}"),
                "confidence": round(float(box.conf.item()), 4),
                "box_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                "box_percent": [
                    round(float(box.xyxy[0][0]) / image_width * 100, 2),
                    round(float(box.xyxy[0][1]) / image_height * 100, 2),
                    round(float(box.xyxy[0][2]) / image_width * 100, 2),
                    round(float(box.xyxy[0][3]) / image_height * 100, 2),
                ],
            })
    return detections


@app.get("/api/health")
def health() -> dict[str, Any]:
    detector = detector_status()
    ollama = ollama_status()
    return {
        "status": "ok",
        "detector": detector["configured"] and all(detector["dependencies"].values()),
        "detector_status": detector,
        "ollama": ollama["connected"] and ollama["model_available"],
        "ollama_status": ollama,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/history")
def history() -> dict[str, Any]:
    return {"items": _history}


@app.get("/api/images/{filename}")
def image(filename: str) -> FileResponse:
    candidate = (OUTPUTS / filename).resolve()
    if candidate.parent != OUTPUTS.resolve() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(candidate)


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)) -> dict[str, Any]:
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="Only JPG, PNG, and WebP images are supported")
    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be between 1 byte and 20 MB")
    inspection_id = f"INS-{datetime.now().strftime('%y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    source_path = UPLOADS / f"{inspection_id}_{Path(file.filename or 'image').name}"
    output_name = f"{inspection_id}_annotated.jpg"
    output_path = OUTPUTS / output_name
    source_path.write_bytes(image_bytes)
    try:
        detections = run_detector(source_path, output_path)
    except Exception as exc:
        source_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"Detector inference failed: {exc}") from exc
    report, llm = ask_ollama(image_bytes, detections)
    item = {
        "inspection_id": inspection_id,
        "filename": file.filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detections": detections,
        "annotated_image_url": f"/api/images/{output_name}",
        "report": report,
        "llm": llm,
        "verified": False,
    }
    _history.insert(0, item)
    del _history[50:]
    return item
