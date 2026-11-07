"""Tile embeddings and the reference gallery they are matched against.

Tiles are embedded with a frozen DINOv2 ViT-S/14, a self-supervised vision backbone whose
features are known to be robust to pose and background changes. Two interchangeable backends
produce the same features:

  * TorchEmbedder (timm + PyTorch) for building galleries and training, `uv sync --extra train`;
  * OnnxEmbedder (onnxruntime) for inference, so a deployment doesn't carry PyTorch. The model is
    exported from the torch one with `recognizer export-onnx`.

The gallery holds embeddings of synthetic tiles rendered from every sprite class's reference
images (see synth.py); without a trained head, a query tile's score for a class is its best
cosine similarity to that class's gallery entries.
"""

from __future__ import annotations

import cv2
import numpy as np

from .paths import CACHE, MODELS

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
INPUT_PX = 224
ONNX_FILE = MODELS / "dinov2_vits14.onnx"
GALLERY_FILE = CACHE / "gallery.npz"
GALLERY_PER_CLASS = 48  # room for both GO art and the other references at full strength
# ImageNet normalization, which DINOv2 was trained with.
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def prepare(tiles: list[np.ndarray]) -> np.ndarray:
    """RGB uint8 tiles -> normalized float32 NCHW batch. Shared by both backends so they see
    identical inputs."""
    batch = np.stack([cv2.resize(t, (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_AREA) for t in tiles])
    x = batch.astype(np.float32).transpose(0, 3, 1, 2) / 255
    return (x - MEAN) / STD


def _normalize(feats: np.ndarray) -> np.ndarray:
    return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)


def device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchEmbedder:
    dim = 384

    def __init__(self):
        import timm

        self.device = device()
        self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_PX)
        self.model.eval().to(self.device)

    def __call__(self, tiles: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """L2-normalized embeddings, one row per RGB tile."""
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(tiles), batch_size):
                x = torch.from_numpy(prepare(tiles[i : i + batch_size])).to(self.device)
                out.append(self.model(x).cpu().numpy())
        return _normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def export_onnx(self, path=ONNX_FILE) -> None:
        import torch

        model = self.model.to("cpu").eval()
        dummy = torch.from_numpy(prepare([np.zeros((INPUT_PX, INPUT_PX, 3), np.uint8)] * 2))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["pixels"],
            output_names=["features"],
            dynamic_axes={"pixels": {0: "batch"}, "features": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        model.to(self.device)


class OnnxEmbedder:
    dim = 384

    def __init__(self, path=ONNX_FILE):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def __call__(self, tiles: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(tiles), batch_size):
            out.append(self.session.run(None, {"pixels": prepare(tiles[i : i + batch_size])})[0])
        return _normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)


# Training code builds galleries and datasets with the torch model.
Embedder = TorchEmbedder


def inference_embedder():
    """The ONNX model when it has been exported, else the torch one."""
    return OnnxEmbedder() if ONNX_FILE.exists() else TorchEmbedder()


def build_gallery(embedder, per_class: int = GALLERY_PER_CLASS, seed: int = 0) -> None:
    from tqdm import tqdm

    from .refs import load_reference_images, sprite_classes
    from .synth import load_backgrounds, load_palette

    rng = np.random.default_rng(seed)
    palette = load_palette()
    backgrounds = load_backgrounds()
    feats, labels = [], []
    for sid in tqdm(sorted(sprite_classes()), desc="gallery", mininterval=10):
        refs = load_reference_images(sid)
        if not refs:
            continue
        tiles = class_tiles(sid, refs, per_class, rng, palette, backgrounds)
        feats.append(embedder(tiles).astype(np.float16))
        labels.extend([sid] * len(tiles))
    GALLERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GALLERY_FILE, feats=np.concatenate(feats), labels=np.array(labels, np.int32))


def class_tiles(sid: int, refs, per_class: int, rng, palette, backgrounds) -> list[np.ndarray]:
    """A class's synthetic tiles: `per_class` cycling through its references, plus its plain GO
    art in every state at set opacities and smoke strengths (synth.PLAIN_GRID)."""
    from . import go_art
    from .synth import plain_variants, synth_tiles

    tiles = synth_tiles(refs, per_class, rng, palette, backgrounds)
    plain = go_art.plain_image(sid)
    if plain is not None:
        tiles += plain_variants(plain, rng, palette, backgrounds)
    return tiles


def rebuild_gallery_classes(embedder, sids: list[int], per_class: int, seed: int = 0) -> None:
    """Re-render the gallery entries of some sprite classes (e.g. after adding extra
    references) and leave every other class's entries untouched."""
    from .refs import load_reference_images
    from .synth import load_backgrounds, load_palette

    data = np.load(GALLERY_FILE)
    keep = ~np.isin(data["labels"], sids)
    feats, labels = [data["feats"][keep]], [data["labels"][keep]]
    rng = np.random.default_rng(seed)
    palette, backgrounds = load_palette(), load_backgrounds()
    for sid in sids:
        refs = load_reference_images(sid)
        tiles = class_tiles(sid, refs, per_class, rng, palette, backgrounds)
        feats.append(embedder(tiles).astype(np.float16))
        labels.append(np.full(len(tiles), sid, np.int32))
    np.savez(GALLERY_FILE, feats=np.concatenate(feats), labels=np.concatenate(labels))


class Gallery:
    def __init__(self, path=GALLERY_FILE):
        from .refs import sprite_classes

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
