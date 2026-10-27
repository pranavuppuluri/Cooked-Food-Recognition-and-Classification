#!/usr/bin/env python3
"""
HealthNow - serves the Indian-food classifier (the DualTrackModel from this
repo) behind a small web page. A photo taken or picked on a phone is POSTed to
/predict; the model returns its top guesses. Every real request is logged under
live_usage/ so the repo accumulates genuine in-the-wild examples.

    WEIGHTS=runs/baseline/best.pth PORT=8000 \
        /mnt/drive/healthnow/venv/bin/python serve/app.py

Runs on CPU, loads the weights once at startup, and needs no internet.
"""
from __future__ import annotations

import datetime
import io
import json
import os
import pathlib
import sys
import threading

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
WEIGHTS = os.environ.get("WEIGHTS", str(REPO / "runs" / "baseline" / "best.pth"))
TRAIN_DIR = os.environ.get("TRAIN_DIR", str(REPO / "ind_food_37" / "train"))
USAGE_DIR = pathlib.Path(os.environ.get("USAGE_DIR", str(REPO / "live_usage")))
PORT = int(os.environ.get("PORT", 8000))

# Reuse the exact model and preprocessing the repo trained with.
sys.path.insert(0, str(REPO / "src"))
from indfood.data import eval_transform  # noqa: E402
from indfood.model import load_model  # noqa: E402

DEVICE = torch.device("cpu")
CLASSES = sorted(p.name for p in pathlib.Path(TRAIN_DIR).iterdir() if p.is_dir())
MODEL = load_model(WEIGHTS, len(CLASSES), DEVICE)
TRANSFORM = eval_transform(224)
_infer_lock = threading.Lock()
_log_lock = threading.Lock()

USAGE_IMAGES = USAGE_DIR / "images"
USAGE_IMAGES.mkdir(parents=True, exist_ok=True)
USAGE_LOG = USAGE_DIR / "predictions.jsonl"

app = Flask(__name__, static_folder=None)
CORS(app)  # read-only classifier; the launcher and the page may call it from any origin


@torch.no_grad()
def classify(img: Image.Image, top: int = 5) -> list[dict]:
    batch = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with _infer_lock:
        probs = MODEL(batch).softmax(1)[0].cpu()
    values, indices = probs.topk(min(top, len(CLASSES)))
    return [{"dish": CLASSES[int(i)], "prob": float(v)} for v, i in zip(values, indices)]


def log_usage(raw: bytes, filename: str, results: list[dict]) -> None:
    """Keep the photo and its prediction so the repo gathers real usage."""
    try:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        ext = (os.path.splitext(filename or "")[1] or ".jpg").lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        name = f"{ts}{ext}"
        (USAGE_IMAGES / name).write_bytes(raw)
        record = {"ts": ts, "image": f"images/{name}", "top": results}
        with _log_lock, open(USAGE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # logging must never break a prediction
        print(f"usage log failed: {exc}", file=sys.stderr)


@app.get("/health")
def health():
    return jsonify(ok=True, model=os.path.basename(WEIGHTS), classes=len(CLASSES))


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Send an image in the 'image' field."), 400
    raw = f.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return jsonify(error="That file is not a readable image."), 400
    results = classify(img)
    log_usage(raw, f.filename or "", results)
    return jsonify(top=results, model=os.path.basename(WEIGHTS))


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


if __name__ == "__main__":
    print(f"HealthNow: {os.path.basename(WEIGHTS)} loaded, {len(CLASSES)} dishes, port {PORT}")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
