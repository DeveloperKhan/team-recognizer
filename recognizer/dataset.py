"""Real labeled tiles, cut from screenshots whose teams are known.

Rosters say which six species a team has but not which tile shows which, so each team's tiles are
matched to its roster species with the assignment that best fits the gallery scores (restricted
to those six species, this is reliable even where the open-set prediction is wrong).

The tiles (saved under cache/tiles with their embeddings) train the shadow classifier and the
species head; `event` lets callers hold events out for testing.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from .assign import min_cost_assignment

from .evaluate import SCREENS_FILE, _truth_team, load_rosters
from .grid import locate
from .io import find_images, load_rgb
from .paths import CACHE
from .pipeline import crop_tile
from .shadow import band_only

TILES_DIR = CACHE / "tiles"
MANIFEST = TILES_DIR / "manifest.json"
FEATURES = TILES_DIR / "features.npz"


@dataclass
class TileRecord:
    event: str
    screen: str
    side: str
    index: int
    sid: int
    shadow: bool
    file: str


def build(screens_dir: Path, embedder, gallery, exclude_events: tuple[str, ...] = ()) -> list[TileRecord]:
    labels = json.loads(SCREENS_FILE.read_text())
    rosters = load_rosters()
    records: list[TileRecord] = []
    tiles_all: list[np.ndarray] = []
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    col = {int(c): i for i, c in enumerate(gallery.classes)}

    for path in find_images(screens_dir):
        key = unicodedata.normalize("NFC", str(path.relative_to(screens_dir)))
        label = labels.get(key)
        if not label or label.get("event") in exclude_events:
            continue
        img = load_rgb(path)
        grids = locate(img)
        for grid in grids:
            side = grid.side if grid.side in label else ("team" if "team" in label and len(grids) == 1 else None)
            if side is None:
                continue
            _, truth = _truth_team(label[side], label.get("event"), rosters)
            if not truth or any(t[0] is None or t[0] not in col for t in truth):
                continue
            tiles = [crop_tile(img, b) for b in grid.tiles()]
            scores = gallery.class_scores(embedder(tiles))[:, [col[t[0]] for t in truth]]
            for r, c in enumerate(min_cost_assignment(-scores)):
                name = f"{label.get('event')}_{path.stem}_{side}_{r}.png".replace(" ", "_")
                cv2.imwrite(str(TILES_DIR / name), cv2.cvtColor(tiles[r], cv2.COLOR_RGB2BGR))
                tiles_all.append(tiles[r])
                records.append(
                    TileRecord(
                        event=label.get("event") or "",
                        screen=key,
                        side=side,
                        index=int(r),
                        sid=int(truth[c][0]),
                        shadow=bool(truth[c][1]),
                        file=name,
                    )
                )
    MANIFEST.write_text(json.dumps([asdict(r) for r in records], indent=1))
    np.savez(FEATURES, tile=embedder(tiles_all), band=embedder([band_only(t) for t in tiles_all]))
    return records


def load() -> tuple[list[TileRecord], dict[str, np.ndarray]]:
    records = [TileRecord(**r) for r in json.loads(MANIFEST.read_text())]
    return records, dict(np.load(FEATURES))


def image(record: TileRecord) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(TILES_DIR / record.file)), cv2.COLOR_BGR2RGB)
