"""Reference sprites: download and decode the images each species is matched against.

Two sources per sprite class (keyed by the Dracoviz `sid`):
  * Showdown's animated 3D sprite, sampled to a handful of frames, which covers the poses the
    overlay's animated sprites pass through;
  * the static Pokémon HOME render the Dracoviz site already serves.
"""

from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageSequence
from tqdm import tqdm

from .paths import CACHE, DATA

SHOWDOWN_ANI = "https://play.pokemonshowdown.com/sprites/ani/{slug}.gif"
HOME_RENDER = "https://imagedelivery.net/2qzpDFW7Yl3NqBaOSqtWxQ/home_{sid}.png/public"
REFS = CACHE / "refs"
FRAMES_PER_GIF = 8


@dataclass(frozen=True)
class Species:
    speciesId: str
    speciesName: str
    dex: int
    sid: int
    showdownSlug: str | None
    shadowEligible: bool


# Forms that can't be brought to a tournament team.
EXCLUDED_FORMS = re.compile(r"_(mega|mega_x|mega_y|primal)$")


def load_species() -> list[Species]:
    return [
        Species(**s)
        for s in json.loads((DATA / "species.json").read_text())
        if not EXCLUDED_FORMS.search(s["speciesId"])
    ]


def sprite_classes() -> dict[int, list[Species]]:
    """sid -> the speciesIds drawn with that sprite (cosmetic forms share one)."""
    classes: dict[int, list[Species]] = {}
    for s in load_species():
        classes.setdefault(s.sid, []).append(s)
    return classes


def _download(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists():
        return dest.stat().st_size > 0
    try:
        res = session.get(url, timeout=30)
    except requests.RequestException:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    # An empty file records a miss so reruns don't refetch it.
    dest.write_bytes(res.content if res.ok else b"")
    return res.ok


def download_all(workers: int = 8) -> None:
    session = requests.Session()
    jobs = []
    for sid, members in sprite_classes().items():
        slug = next((m.showdownSlug for m in members if m.showdownSlug), None)
        jobs.append((HOME_RENDER.format(sid=sid), REFS / "home" / f"{sid}.png"))
        if slug:
            jobs.append((SHOWDOWN_ANI.format(slug=slug), REFS / "showdown" / f"{sid}.gif"))
    with ThreadPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(lambda j: _download(*j, session), jobs), total=len(jobs), desc="sprites"))
    missing = [str(dest.relative_to(REFS)) for (_, dest), ok in zip(jobs, results) if not ok]
    if missing:
        print(f"{len(missing)} sprites unavailable: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")


def _trim(rgba: np.ndarray) -> np.ndarray | None:
    """Crop to the opaque bounding box; None if the image is empty."""
    ys, xs = np.nonzero(rgba[..., 3] > 16)
    if len(ys) == 0:
        return None
    return rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def load_reference_images(sid: int) -> list[np.ndarray]:
    """All RGBA reference images for a sprite class, trimmed to their content."""
    images = []
    home = REFS / "home" / f"{sid}.png"
    if home.exists() and home.stat().st_size:
        img = Image.open(home).convert("RGBA")
        img.thumbnail((256, 256), Image.LANCZOS)
        images.append(np.asarray(img))
    gif = REFS / "showdown" / f"{sid}.gif"
    if gif.exists() and gif.stat().st_size:
        frames = [f.convert("RGBA") for f in ImageSequence.Iterator(Image.open(io.BytesIO(gif.read_bytes())))]
        picks = np.linspace(0, len(frames) - 1, min(FRAMES_PER_GIF, len(frames))).round().astype(int)
        images.extend(np.asarray(frames[i]) for i in sorted(set(picks)))
    return [t for t in (_trim(i) for i in images) if t is not None]
