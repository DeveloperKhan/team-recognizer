"""Shadow detection: shadow Pokémon sit on tiles mottled with purple smoke.

Two checks on the tile's border band (where sprites rarely reach):

  * Color: the fraction of violet pixels. Nothing else on the overlay is violet, so when the smoke
    keeps its color this is all it takes (on full broadcast frames it is right on essentially
    every tile, and it fires on ~1 in 270 non-shadow tiles).
  * A learned probe for when it doesn't: heavily compressed crops wash the smoke out to a slightly
    bluer teal, and fainted tiles are gray. The band (center blanked out) is embedded with the
    same DINOv2 model used for species and a logistic regression, trained on labeled real tiles,
    separates smoke from fill.

A tile is shadow if the color check fires or the probe is confident.
"""

from __future__ import annotations

import numpy as np

from .paths import MODELS

MODEL_FILE = MODELS / "shadow.npz"
BAND = 0.18  # border band kept, as a fraction of the tile
COLOR_BAND = 0.16  # border band the violet check looks at
COLOR_THRESHOLD = 0.03  # violet fraction of the band that marks smoke
PROBE_THRESHOLD = 0.7


def violet_fraction(tile: np.ndarray) -> float:
    """Share of the border band (rim excluded) that is violet: hue 108-165 on OpenCV's 0-180 scale,
    with enough saturation and brightness to rule out gray and shadowed pixels."""
    import cv2

    h, w = tile.shape[:2]
    bh, bw = max(1, int(COLOR_BAND * h)), max(1, int(COLOR_BAND * w))
    rim = max(1, int(0.03 * h))
    mask = np.zeros((h, w), bool)
    mask[:bh] = mask[-bh:] = True
    mask[:, :bw] = mask[:, -bw:] = True
    mask[:rim] = mask[-rim:] = False
    mask[:, :rim] = mask[:, -rim:] = False
    H, S, V = cv2.cvtColor(tile[mask][None], cv2.COLOR_RGB2HSV)[0].astype(int).T
    return float(np.mean((H >= 108) & (H <= 165) & (S >= 35) & (V >= 35)))


def band_only(tile: np.ndarray) -> np.ndarray:
    """The tile with everything but its border band painted over in the band's median color."""
    h, w = tile.shape[:2]
    bh, bw = max(1, int(BAND * h)), max(1, int(BAND * w))
    band = np.concatenate(
        [tile[:bh].reshape(-1, 3), tile[-bh:].reshape(-1, 3), tile[:, :bw].reshape(-1, 3), tile[:, -bw:].reshape(-1, 3)]
    )
    out = tile.copy()
    out[bh : h - bh, bw : w - bw] = np.median(band, 0)
    return out


class ShadowClassifier:
    def __init__(self, embedder):
        self.embedder = embedder
        self.model = dict(np.load(MODEL_FILE)) if MODEL_FILE.exists() else None

    @property
    def trained(self) -> bool:
        return self.model is not None

    def classify(self, tiles: list[np.ndarray]) -> list[dict]:
        """Per tile: the decision plus both signals behind it."""
        probe = (
            predict(self.model, self.embedder([band_only(t) for t in tiles]))
            if self.model is not None
            else np.zeros(len(tiles))
        )
        out = []
        for tile, p in zip(tiles, probe):
            violet = violet_fraction(tile)
            out.append(
                {
                    "shadow": bool(violet >= COLOR_THRESHOLD or p >= PROBE_THRESHOLD),
                    "violet": round(violet, 3),
                    "probe": round(float(p), 3),
                }
            )
        return out


def predict(model: dict, feats: np.ndarray) -> np.ndarray:
    z = ((feats - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"]
    return 1 / (1 + np.exp(-z))


def fit(feats: np.ndarray, labels: np.ndarray, C: float = 0.1, save: bool = True) -> dict:
    """Fit on band embeddings of labeled tiles (and save, unless experimenting)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(feats)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(scaler.transform(feats), labels)
    model = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "coef": clf.coef_[0].astype(np.float32),
        "intercept": np.float32(clf.intercept_[0]),
    }
    if save:
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(MODEL_FILE, **model)
    return model
