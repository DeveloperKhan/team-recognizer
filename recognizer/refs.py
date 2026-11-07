"""Reference sprites: download and decode the images each species is matched against.

Two kinds of reference per sprite class (keyed by the Dracoviz `sid`):
  * Pokémon GO's own art (go_art.py: the GO wiki's sprites and PokeMiners' mined icons), in each
    form, costume and shiny the class covers: what the overlay actually draws, so it makes up half
    of a class's references;
  * Showdown's animated 3D sprite, sampled to a handful of frames, which covers the poses the
    overlay's animated sprites pass through.
HOME renders were dropped: their poses often differ from the overlay's (Stunfisk face-on).
"""

from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from .paths import CACHE, DATA, EXTRA_REFS

SHOWDOWN_ANI = "https://play.pokemonshowdown.com/sprites/ani/{slug}.gif"
REFS = CACHE / "refs"
# Extra reference sprites live in your data directory (paths.EXTRA_REFS):
# refs/extra/<speciesId>/<any name>.{png,gif,webp}, for overlay art the downloaded references don't
# cover, e.g. the white-flowered Florges the 2027 overlay shows (Showdown's Florges is red).
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


def _download(url: str, dest: Path, session) -> bool:
    import requests

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
    import requests
    from tqdm import tqdm

    from . import go_art

    counts = go_art.download(sprite_classes(), workers)
    print(f"GO art: {counts['icons']} icons for {counts['classes']} classes ({counts['failed']} unavailable)")

    session = requests.Session()
    jobs = []
    for sid, members in sprite_classes().items():
        slug = next((m.showdownSlug for m in members if m.showdownSlug), None)
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


def _frames(path: Path) -> list[np.ndarray]:
    """Up to FRAMES_PER_GIF evenly spaced frames of an image file (one for a still image)."""
    frames = [f.convert("RGBA") for f in ImageSequence.Iterator(Image.open(io.BytesIO(path.read_bytes())))]
    picks = np.linspace(0, len(frames) - 1, min(FRAMES_PER_GIF, len(frames))).round().astype(int)
    out = []
    for i in sorted(set(picks)):
        frame = frames[i]
        frame.thumbnail((256, 256), Image.LANCZOS)
        out.append(np.asarray(frame))
    return out


def extra_reference_images(sid: int) -> list[np.ndarray]:
    """Frames of the extra references for any speciesId drawn with this sprite."""
    images = []
    for species in sprite_classes().get(sid, []):
        folder = EXTRA_REFS / species.speciesId
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in {".png", ".gif", ".webp"}:
                    images.extend(_frames(path))
    return [t for t in (_trim(i) for i in images) if t is not None]


# Share of a class's references that are GO art, when it has some. Half: GO's icon is how the
# overlay draws a Pokémon, the other art covers the poses its animation passes through.
GO_SHARE = 1 / 2


def _mix(primary: list, secondary: list, share: float) -> list:
    """`primary` repeated to make up `share` of the result, spread evenly through `secondary`,
    since synthetic tiles cycle through the list from the start."""
    if not primary or not secondary:
        return primary or secondary
    n = max(len(primary), round(len(secondary) * share / (1 - share)))
    first = [primary[i % len(primary)] for i in range(n)]
    placed = [((k + 0.5) / n, 0, k) for k in range(n)] + [
        ((k + 0.5) / len(secondary), 1, k) for k in range(len(secondary))
    ]
    return [(first if which == 0 else secondary)[k] for _, which, k in sorted(placed)]


def load_reference_images(sid: int) -> list[np.ndarray]:
    """All RGBA reference images for a sprite class, trimmed to their content.

    GO art makes up half of the list where a class has some, and extra references about half of
    the rest, since synthetic tiles cycle through it: both exist because the other art doesn't
    match the overlay.
    """
    from . import go_art

    images = []
    gif = REFS / "showdown" / f"{sid}.gif"
    if gif.exists() and gif.stat().st_size:
        images.extend(_frames(gif))
    base = [t for t in (_trim(i) for i in images) if t is not None]
    go = [t for t in (_trim(i) for i in go_art.reference_images(sid)) if t is not None]
    base = _mix(go, base, GO_SHARE)
    extra = extra_reference_images(sid)
    if not extra:
        return base
    repeat = max(1, round(len(base) / len(extra)))
    # Interleave so a class's first synthetic tiles already include both looks.
    mixed = []
    for i in range(max(len(base), len(extra) * repeat)):
        if i < len(base):
            mixed.append(base[i])
        mixed.append(extra[i % len(extra)])
    return mixed
