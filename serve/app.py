#!/usr/bin/env python3
"""
HealthNow - serves the Indian-food classifier (the DualTrackModel from this
repo) behind a small web page. A photo taken or picked on a phone is POSTed to
/predict; the model returns its top guesses. Every real request is logged under
live_usage/ so the repo accumulates genuine in-the-wild examples.

    WEIGHTS=runs/baseline/best.pth PORT=8000 \
        /mnt/drive/healthnow/venv/bin/python serve/app.py

Runs on CPU, loads the weights once at startup, and needs no internet.

The page is open to anyone, but after HEALTHNOW_FREE_USES free photos a browser
session must send the password in HEALTHNOW_PASSWORD to keep classifying. Both
come from the environment, never from this file, so the password stays out of
the repo. With HEALTHNOW_PASSWORD unset the gate is off and the page is fully
open.
"""
from __future__ import annotations

import datetime
import io
import json
import os
import pathlib
import secrets
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

# Soft gate: the first FREE_USES photos in a browser session are free, then a
# password is required. The password lives in the environment, not the repo.
PASSWORD = os.environ.get("HEALTHNOW_PASSWORD", "")
FREE_USES = int(os.environ.get("HEALTHNOW_FREE_USES", "1"))
COOKIE = "hn_sid"

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

# Per-session trial counters, keyed by the hn_sid cookie. In memory only: a
# restart resets the gate, which is fine for a personal demo.
SESSIONS: dict[str, dict] = {}
_sess_lock = threading.Lock()

USAGE_IMAGES = USAGE_DIR / "images"
USAGE_IMAGES.mkdir(parents=True, exist_ok=True)
USAGE_LOG = USAGE_DIR / "predictions.jsonl"

app = Flask(__name__, static_folder=None)
CORS(app)  # read-only classifier; the page that calls it is served from here


def get_session() -> tuple[str, dict]:
    """Find or start the caller's trial session from the hn_sid cookie."""
    sid = request.cookies.get(COOKIE)
    with _sess_lock:
        state = SESSIONS.get(sid) if sid else None
        if state is None:
            sid = secrets.token_urlsafe(16)
            state = {"uses": 0, "unlocked": False}
            SESSIONS[sid] = state
            # a public URL could accrue sessions forever; cap the table
            if len(SESSIONS) > 5000:
                for stale in list(SESSIONS)[:1000]:
                    SESSIONS.pop(stale, None)
        return sid, state


def _with_cookie(resp, sid: str, status: int = 200):
    resp.status_code = status
    resp.set_cookie(COOKIE, sid, max_age=60 * 60 * 24 * 365,
                    httponly=True, secure=True, samesite="Lax")
    return resp


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
    sid, state = get_session()
    gate_on = bool(PASSWORD)
    locked = gate_on and not state["unlocked"] and state["uses"] >= FREE_USES

    if locked:
        given = (request.form.get("password") or "").strip()
        if given and secrets.compare_digest(given, PASSWORD):
            state["unlocked"] = True
            locked = False
        else:
            resp = jsonify(error="password_required",
                           message="Enter the password to keep using HealthNow.")
            return _with_cookie(resp, sid, 401)

    f = request.files.get("image")
    if f is None:
        return _with_cookie(jsonify(error="Send an image in the 'image' field."), sid, 400)
    raw = f.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return _with_cookie(jsonify(error="That file is not a readable image."), sid, 400)

    results = classify(img)
    log_usage(raw, f.filename or "", results)
    if gate_on and not state["unlocked"]:
        state["uses"] += 1

    free_left = None
    if gate_on and not state["unlocked"]:
        free_left = max(0, FREE_USES - state["uses"])
    resp = jsonify(top=results, model=os.path.basename(WEIGHTS),
                   free_left=free_left, unlocked=state["unlocked"])
    return _with_cookie(resp, sid)


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


if __name__ == "__main__":
    gate = f"password after {FREE_USES} free" if PASSWORD else "open (no password set)"
    print(f"HealthNow: {os.path.basename(WEIGHTS)} loaded, {len(CLASSES)} dishes, port {PORT}, gate {gate}")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
