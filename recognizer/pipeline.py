"""Screenshot in, teams out."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import forms, train_head
from .assign import assign_team
from .embed import Gallery, inference_embedder
from .grid import Grid, locate
from .refs import sprite_classes
from .shadow import ShadowClassifier

TILE_INSET = 0.02  # trim the tile border itself before embedding


@dataclass
class TeamResult:
    side: str
    grid: Grid
    player: str | None = None
    player_raw: str | None = None
    country: str | None = None
    pokemon: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "player": self.player,
            "player_raw": self.player_raw,
            "country": self.country,
            **{k: v for k, v in self.grid.to_dict().items() if k != "side"},
            "pokemon": self.pokemon,
        }


def crop_tile(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * TILE_INSET, (y1 - y0) * TILE_INSET
    H, W = img.shape[:2]
    xa, ya = int(round(max(0, x0 + dx))), int(round(max(0, y0 + dy)))
    xb, yb = int(round(min(W, x1 - dx))), int(round(min(H, y1 - dy)))
    if xb - xa < 4 or yb - ya < 4:
        return np.zeros((8, 8, 3), np.uint8)
    return img[ya:yb, xa:xb]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class SpeciesScorer:
    """Per-class scores for tile embeddings: the trained head if there is one, otherwise
    nearest-neighbour similarity to the synthetic gallery. Forms that look the same on the
    overlay are merged into one class (see forms.py)."""

    def __init__(self):
        head = train_head.load()
        if head is not None:
            self.weights, self.classes = head
            self.gallery = None
        else:
            self.gallery = Gallery()
            self.classes = self.gallery.classes
        species = {sid: min(m, key=lambda s: len(s.speciesId)).speciesId for sid, m in sprite_classes().items()}
        self.merge = forms.merge_plan([species[int(sid)] for sid in self.classes])

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        """(n, n_classes) log-probabilities."""
        if self.gallery is None:
            logits = train_head.SCALE * feats @ self.weights.T
        else:
            logits = 30 * self.gallery.class_scores(feats)
        probs = forms.merge_probabilities(_softmax(logits), self.merge)
        return np.log(probs + 1e-12)


class Recognizer:
    def __init__(self, read_names: bool = True):
        self.embedder = inference_embedder()
        self.species = SpeciesScorer()
        self.shadow = ShadowClassifier(self.embedder)
        self.read_names = read_names
        # One speciesId per sprite: cosmetic forms share a sprite, report the plainest id.
        self.species_for_sid = {sid: min(m, key=lambda s: len(s.speciesId)) for sid, m in sprite_classes().items()}

    def classify_tiles(self, tiles: list[np.ndarray]) -> list[dict]:
        """Tiles on their own, without a team (so without the assignment across six): each
        one's species, shadow and confidence."""
        if not tiles:
            return []
        logp = self.species(self.embedder(tiles))
        out = []
        for row, sh in zip(logp, self.shadow.classify(tiles)):
            pick = int(np.argmax(row))
            species = self.species_for_sid[int(self.species.classes[pick])]
            out.append(
                {
                    "speciesId": species.speciesId,
                    "shadow": bool(sh["shadow"] and species.shadowEligible),
                    "confidence": round(float(np.exp(row[pick])), 3),
                }
            )
        return out

    def recognize(self, img: np.ndarray, roster_names: list[str] | None = None) -> list[TeamResult]:
        from . import ocr

        teams = []
        for grid in locate(img):
            boxes = grid.tiles()
            tiles = [crop_tile(img, b) for b in boxes]
            logp = self.species(self.embedder(tiles))
            picks = assign_team(logp)
            shadows = self.shadow.classify(tiles)
            team = TeamResult(side=grid.side, grid=grid)
            for box, row, pick, sh in zip(boxes, logp, picks, shadows):
                species = self.species_for_sid[int(self.species.classes[pick])]
                shadow = sh["shadow"] and species.shadowEligible
                team.pokemon.append(
                    {
                        "speciesId": f"{species.speciesId}_shadow" if shadow else species.speciesId,
                        "shadow": shadow,
                        "confidence": round(float(np.exp(row[pick])), 3),
                        "shadow_violet": sh["violet"],
                        "shadow_probe": sh["probe"],
                        "top3": [
                            [
                                self.species_for_sid[int(self.species.classes[i])].speciesId,
                                round(float(np.exp(row[i])), 3),
                            ]
                            for i in np.argsort(-row)[:3]
                        ],
                        "box": [round(v, 1) for v in box],
                    }
                )
            if self.read_names:
                team.player_raw = ocr.read_name(img, grid.name_bar())
                team.country = ocr.read_country(img, grid.country_tag())
                team.player, _ = ocr.match_name(team.player_raw, roster_names or [])
                if team.player is None and not roster_names:
                    team.player = team.player_raw
            teams.append(team)
        return teams
