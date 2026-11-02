"""HTTP API for the recognizer (deployed to Vercel as a FastAPI function).

    POST /recognize   body: the image (multipart field "image", or the raw image bytes)
                      header: Authorization: Bearer $RECOGNIZER_TOKEN (when that env var is set)
    GET  /health

Run locally with `uv run --extra train uvicorn server:app --port 8000`.
"""

from __future__ import annotations

import io
import os
import secrets
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

from recognizer.pipeline import Recognizer

MAX_SIDE = 2400  # larger uploads are scaled down; the grid search doesn't need more

app = FastAPI(title="team-recognizer")
_recognizer: Recognizer | None = None


def recognizer() -> Recognizer:
    # Loaded on first use so a cold start only pays for it when a request needs it.
    global _recognizer
    if _recognizer is None:
        _recognizer = Recognizer()
    return _recognizer


def _authorize(request: Request) -> None:
    token = os.environ.get("RECOGNIZER_TOKEN")
    if not token:
        return  # local development
    sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(sent, token):
        raise HTTPException(status_code=401, detail="invalid token")


async def _read_image(request: Request) -> np.ndarray:
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=400, detail='expected a file in the "image" field')
        data = await upload.read()
    else:
        data = await request.body()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="not a readable image") from None
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 255)), img)
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(img)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/recognize")
async def recognize(request: Request) -> dict:
    _authorize(request)
    img = await _read_image(request)
    start = time.perf_counter()
    teams = recognizer().recognize(img)
    return {
        "teams": [
            {
                "side": team.side or None,
                "player": team.player_raw,
                "country": team.country,
                "pokemon": [
                    {
                        "speciesId": p["speciesId"].removesuffix("_shadow") if p["shadow"] else p["speciesId"],
                        "shadow": p["shadow"],
                        "confidence": p["confidence"],
                        "alternatives": [{"speciesId": s, "confidence": c} for s, c in p["top3"][1:]],
                    }
                    for p in team.pokemon
                ],
            }
            for team in teams
        ],
        "elapsedMs": round((time.perf_counter() - start) * 1000),
    }
