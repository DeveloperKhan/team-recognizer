"""Player names from the name bar above each grid (RapidOCR: PaddleOCR's detector and CRNN
recognizer run through onnxruntime)."""

from __future__ import annotations

import re

import cv2
import numpy as np
from rapidfuzz import fuzz, process

_engine = None
MATCH_THRESHOLD = 70

# Name bars also hold the match score and a country code; neither is part of the name.
_NOISE = re.compile(r"^(\d+|[A-Z]{2})$")
_RESULT_SUFFIX = re.compile(r"\s*\((W|L)\)\s*$", re.I)


def _ocr():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def read_name(img: np.ndarray, box) -> str | None:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if y1 - y0 < 8 or x1 - x0 < 16:
        return None
    crop = img[y0:y1, x0:x1]
    scale = 96 / crop.shape[0]
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result, _ = _ocr()(crop)
    if not result:
        return None
    texts = []
    for points, text, conf in result:
        text = _RESULT_SUFFIX.sub("", text.strip())
        if conf < 0.5 or not text or _NOISE.match(text):
            continue
        width = max(p[0] for p in points) - min(p[0] for p in points)
        texts.append((width, text))
    if not texts:
        return None
    return max(texts)[1]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_name(raw: str | None, names: list[str]) -> tuple[str | None, float]:
    """Best roster name for an OCR reading (names are upper-cased on some overlays and OCR
    confuses O/0 and I/l, so matching is fuzzy on a normalized key)."""
    if not raw or not names:
        return None, 0.0
    keys = {_key(n): n for n in names}
    hit = process.extractOne(_key(raw), list(keys), scorer=fuzz.ratio)
    if hit and hit[1] >= MATCH_THRESHOLD:
        return keys[hit[0]], float(hit[1])
    return None, float(hit[1]) if hit else 0.0
