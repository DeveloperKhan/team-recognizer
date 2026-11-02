"""Player names from the name bar above each grid (RapidOCR: PaddleOCR's detector and CRNN
recognizer run through onnxruntime)."""

from __future__ import annotations

import re

import cv2
import numpy as np
from rapidfuzz import fuzz, process

_engine = None
MATCH_THRESHOLD = 70

# ISO 3166-1 alpha-2 codes, as play-graphic-tool's flag picker lists them.
COUNTRY_CODES = frozenset(
    """
AD AE AF AG AI AL AM AO AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ
BL BM BN BO BQ BR BS BT BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR
CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO
FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GT GU GW GY HK HN HR HT HU
ID IE IL IM IN IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ
LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP
MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA
PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE
SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TG TH TJ TK TL TM
TN TO TR TT TV TW TZ UA UG US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA
ZM ZW
    """.split()
)

# Name bars also hold the match score and a country code; neither is part of the name.
_NOISE = re.compile(r"^(\d+|[A-Z]{2})$")
_RESULT_SUFFIX = re.compile(r"\s*\((W|L)\)\s*$", re.I)


def _ocr():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def _read(img: np.ndarray, box, height: int = 96):
    """RapidOCR results for a region of the image, or [] if the region is off-image."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if y1 - y0 < 8 or x1 - x0 < 16:
        return []
    crop = img[y0:y1, x0:x1]
    scale = height / crop.shape[0]
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result, _ = _ocr()(crop)
    return result or []


def read_country(img: np.ndarray, box) -> str | None:
    """The country tag shown above the name bar (absent on tight crops)."""
    for _, text, conf in _read(img, box, height=120):
        code = re.sub(r"[^A-Z]", "", text.upper())
        if conf >= 0.6 and code in COUNTRY_CODES:
            return code
    return None


def read_name(img: np.ndarray, box) -> str | None:
    result = _read(img, box)
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
