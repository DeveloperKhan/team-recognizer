"""Tile embeddings and the reference gallery they are matched against.

Tiles are embedded with a frozen DINOv2 ViT-S/14, a self-supervised vision backbone whose
features are known to be robust to pose and background changes. The gallery holds embeddings of
synthetic tiles rendered from every sprite class's reference images (see synth.py); a query
tile's score for a class is its best cosine similarity to that class's gallery entries.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .paths import CACHE
from .refs import load_reference_images, sprite_classes
from .synth import load_backgrounds, load_palette, synth_tiles

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
INPUT_PX = 224
GALLERY_FILE = CACHE / "gallery.npz"
GALLERY_PER_CLASS = 24


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Embedder:
    def __init__(self):
        import timm

        self.device = device()
        self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_PX)
        self.model.eval().to(self.device)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=self.device).view(1, 3, 1, 1)

    def _prepare(self, tiles: list[np.ndarray]) -> torch.Tensor:
        import cv2

        batch = np.stack([cv2.resize(t, (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_AREA) for t in tiles])
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255
        return (x - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, tiles: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """L2-normalized embeddings, one row per RGB tile."""
        out = []
        for i in range(0, len(tiles), batch_size):
            feats = self.model(self._prepare(tiles[i : i + batch_size]))
            out.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.num_features), np.float32)


def build_gallery(embedder: Embedder, per_class: int = GALLERY_PER_CLASS, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    palette = load_palette()
    backgrounds = load_backgrounds()
    feats, labels = [], []
    for sid in tqdm(sorted(sprite_classes()), desc="gallery", mininterval=10):
        refs = load_reference_images(sid)
        if not refs:
            continue
        feats.append(embedder(synth_tiles(refs, per_class, rng, palette, backgrounds)).astype(np.float16))
        labels.extend([sid] * per_class)
    GALLERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GALLERY_FILE, feats=np.concatenate(feats), labels=np.array(labels, np.int32))


class Gallery:
    def __init__(self, path=GALLERY_FILE):
        data = np.load(path)
        keep = np.isin(data["labels"], list(sprite_classes()))
        self.feats = data["feats"][keep].astype(np.float32)
        self.labels = data["labels"][keep]
        self.classes, self.index = np.unique(self.labels, return_inverse=True)

    def class_scores(self, queries: np.ndarray) -> np.ndarray:
        """(n_queries, n_classes): best similarity to any gallery entry of each class."""
        sims = queries @ self.feats.T
        out = np.full((len(queries), len(self.classes)), -1.0, np.float32)
        np.maximum.at(out.T, self.index, sims.T)
        return out
