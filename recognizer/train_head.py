"""A linear (cosine) classifier over frozen DINOv2 features.

Nearest-neighbour matching weighs every feature dimension equally, including the ones that
encode tile fill, smoke and dimming. A classifier trained on the synthetic gallery (plus any real
labeled tiles) learns which directions identify the sprite and which to ignore.
"""

from __future__ import annotations

import numpy as np

from .paths import MODELS

HEAD_FILE = MODELS / "species_head.npz"
SCALE = 30.0  # cosine-logit temperature


def _head_class():
    import torch

    class CosineHead(torch.nn.Module):
        def __init__(self, dim: int, n_classes: int):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(n_classes, dim) * 0.01)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return SCALE * torch.nn.functional.normalize(x, dim=1) @ torch.nn.functional.normalize(self.weight, dim=1).T

    return CosineHead


def train(
    feats: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray | None = None,
    epochs: int = 40,
    seed: int = 0,
):
    import torch

    from .embed import device

    torch.manual_seed(seed)
    dev = device()
    index = {int(c): i for i, c in enumerate(classes)}
    x = torch.from_numpy(feats.astype(np.float32)).to(dev)
    y = torch.tensor([index[int(l)] for l in labels], device=dev)
    w = torch.from_numpy((weights if weights is not None else np.ones(len(labels))).astype(np.float32)).to(dev)
    head = _head_class()(feats.shape[1], len(classes)).to(dev)
    # Start each class at the mean of its examples; training then sharpens the boundaries.
    with torch.no_grad():
        head.weight.zero_().index_add_(0, y, torch.nn.functional.normalize(x, dim=1))
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            b = perm[i : i + 1024]
            # Feature dropout stands in for image augmentation, which can't be done on features.
            xb = torch.nn.functional.dropout(x[b], 0.1)
            loss = (
                torch.nn.functional.cross_entropy(head(xb), y[b], reduction="none", label_smoothing=0.05) * w[b]
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return head.eval()


def save(head, classes: np.ndarray) -> None:
    """Store normalized class weights as plain numpy, so inference doesn't need PyTorch."""
    w = head.weight.detach().cpu().numpy()
    w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
    HEAD_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HEAD_FILE, weight=w.astype(np.float32), classes=classes.astype(np.int64))


def load() -> tuple[np.ndarray, np.ndarray] | None:
    """(normalized class weights, classes), or None if no head has been trained."""
    if not HEAD_FILE.exists():
        return None
    data = np.load(HEAD_FILE)
    return data["weight"], data["classes"]


def scores(head, feats: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        x = torch.from_numpy(feats.astype(np.float32)).to(next(head.parameters()).device)
        return head(x).cpu().numpy()
