"""Render synthetic overlay tiles from reference sprites.

Real tiles show the sprite in one of several states, all of which the gallery has to cover:
  * bright   - brought to this game: full-color sprite on the tile fill;
  * dimmed   - not brought: the sprite fades toward the tile fill (partly transparent);
  * gray     - fainted: the whole tile desaturated;
and any of those can sit on a shadow tile, whose fill is mottled with purple smoke.

Tile fills are sampled from real screenshots (see `harvest_palette`), so each overlay's colors are
represented without hand-tuning.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .paths import CACHE

TILE = 160  # synthetic tile width; height follows the style's aspect ratio
PALETTE_FILE = CACHE / "palette.json"
BACKGROUNDS_DIR = CACHE / "backgrounds"
RING = 0.16  # border band (fraction of the tile) searched for sprite-free patches

# Fallback tile fills (RGB) if no palette has been harvested from screenshots yet.
DEFAULT_FILLS = [(108, 196, 92), (60, 170, 110), (36, 120, 96), (30, 92, 78), (24, 84, 70)]
SMOKE_PURPLE = [(118, 52, 170), (140, 70, 200), (96, 40, 150), (160, 90, 210)]


@dataclass
class TileSpec:
    state: str  # bright | dim | gray
    shadow: bool
    fill: tuple[int, int, int]
    aspect: float
    background: np.ndarray | None = None  # a real, sprite-free tile background (RGB)


def load_palette() -> list[tuple[int, int, int]]:
    if PALETTE_FILE.exists():
        return [tuple(c) for c in json.loads(PALETTE_FILE.read_text())["fills"]]
    return DEFAULT_FILLS


def tile_ring(img: np.ndarray, box, inner: float = 0.12, outer: float = 0.04) -> np.ndarray:
    """Pixels of the band just inside a tile's border, where the fill shows around the sprite."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    H, W = img.shape[:2]
    clip = lambda v, hi: int(min(max(round(v), 0), hi))  # noqa: E731
    ox0, oy0, ox1, oy1 = clip(x0 + outer * w, W), clip(y0 + outer * h, H), clip(x1 - outer * w, W), clip(y1 - outer * h, H)
    ix0, iy0, ix1, iy1 = clip(x0 + inner * w, W), clip(y0 + inner * h, H), clip(x1 - inner * w, W), clip(y1 - inner * h, H)
    mask = np.zeros((H, W), bool)
    mask[oy0:oy1, ox0:ox1] = True
    mask[iy0:iy1, ix0:ix1] = False
    return img[mask]


def harvest_palette(images: list[np.ndarray], k: int = 12) -> list[tuple[int, int, int]]:
    """Cluster the fill colors of every located tile into a palette and save it."""
    from sklearn.cluster import KMeans

    from .grid import locate

    fills = []
    for img in images:
        for grid in locate(img):
            for box in grid.tiles():
                ring = tile_ring(img, box)
                if len(ring) < 20:
                    continue
                lab = cv2.cvtColor(ring[None].astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(int)
                # Skip smoke-covered tiles (purple: a* high, b* low) and fainted (gray) ones.
                med = np.median(lab, 0)
                if med[1] > 140 and med[2] < 120 or abs(med[1] - 128) + abs(med[2] - 128) < 8:
                    continue
                fills.append(np.median(ring, 0))
    centers = KMeans(min(k, len(fills)), n_init=4, random_state=0).fit(np.array(fills)).cluster_centers_
    palette = [tuple(int(round(c)) for c in center) for center in centers]
    PALETTE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PALETTE_FILE.write_text(json.dumps({"fills": palette}))
    return palette


def _noise(h: int, w: int, rng: np.random.Generator, octaves: int = 3) -> np.ndarray:
    """Smooth value noise in [0, 1]."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 3 * 2**o
        grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def smoke_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Blotchy purple-smoke coverage in [0, 1], denser toward the tile edges."""
    n = _noise(h, w, rng, octaves=4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum.reduce([xx / w, 1 - xx / w, yy / h, 1 - yy / h]) * 2  # 0 at edge, 1 at center
    threshold = rng.uniform(0.45, 0.6) + 0.25 * edge
    mask = np.clip((n - threshold) * rng.uniform(8, 20), 0, 1)
    return cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.5, 1.5))


def render_tile(sprite: np.ndarray, spec: TileSpec, rng: np.random.Generator) -> np.ndarray:
    """Composite an RGBA sprite into a tile; returns RGB uint8."""
    w = TILE
    h = int(round(TILE * spec.aspect))
    if spec.background is not None:
        tile = cv2.resize(spec.background, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        fill = np.median(tile.reshape(-1, 3), 0)
    else:
        fill = np.array(spec.fill, np.float32)
        # Fills are close to flat; add a faint vertical gradient and grain.
        grad = np.linspace(1 + rng.uniform(-0.06, 0.06), 1, h, dtype=np.float32)[:, None, None]
        tile = np.clip(fill[None, None] * grad + rng.normal(0, 1.5, (h, w, 3)), 0, 255)

    if spec.shadow and spec.background is None:
        m = smoke_mask(h, w, rng)[..., None]
        purple = np.array(SMOKE_PURPLE[rng.integers(len(SMOKE_PURPLE))], np.float32)
        strength = rng.uniform(0.75, 1.0) if spec.state == "bright" else rng.uniform(0.2, 0.55)
        tile = tile * (1 - m * strength) + purple * m * strength

    # Scale the sprite to fit, with the jitter seen between overlays and animation frames.
    sh, sw = sprite.shape[:2]
    fit = rng.uniform(0.62, 0.92) * min(w / sw, h / sh)
    nw, nh = max(1, int(sw * fit)), max(1, int(sh * fit))
    spr = np.asarray(Image.fromarray(sprite).resize((nw, nh), Image.LANCZOS)).astype(np.float32)
    x0 = int((w - nw) / 2 + rng.uniform(-0.06, 0.06) * w)
    y0 = int((h - nh) / 2 + rng.uniform(-0.05, 0.08) * h)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x0 + nw), min(h, y0 + nh)
    crop = spr[ya - y0 : yb - y0, xa - x0 : xb - x0]
    alpha = crop[..., 3:4] / 255
    rgb = crop[..., :3]
    if spec.state == "dim":
        # Dimmed sprites are see-through (the tile, smoke included, shows behind them) and
        # slightly tinted toward the fill.
        alpha = alpha * rng.uniform(0.3, 0.65)
        tint = rng.uniform(0, 0.3)
        rgb = rgb * (1 - tint) + fill * tint
    region = tile[ya:yb, xa:xb]
    tile[ya:yb, xa:xb] = region * (1 - alpha) + rgb * alpha

    if spec.state == "gray":
        gray = tile.mean(-1, keepdims=True)
        tile = np.repeat(gray, 3, -1) * rng.uniform(0.6, 0.9) + rng.uniform(10, 40)

    return np.clip(tile, 0, 255).astype(np.uint8)


def degrade(tile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stream/screenshot artifacts: blur, low resolution, lossy compression, slight crop."""
    h, w = tile.shape[:2]
    out = tile
    # Grid snapping is off by a few percent at times.
    dx, dy = (int(rng.uniform(-0.04, 0.04) * w) for _ in range(2))
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.7:
        s = rng.uniform(0.35, 0.9)
        small = cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.8:
        buf = io.BytesIO()
        fmt = "WEBP" if rng.random() < 0.5 else "JPEG"
        Image.fromarray(out).save(buf, fmt, quality=int(rng.uniform(30, 90)))
        out = np.asarray(Image.open(buf).convert("RGB"))
    return out


def sprite_free_background(tile: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """A tile's background with the sprite removed, or None if too little of it is clean.

    The center is refilled with patches copied from the tile's border band, which keeps its fill,
    gradient and any smoke pattern. Sprites often reach into that band (wings, tails), so only
    patches made entirely of the band's two dominant colors (fill and smoke) are used.
    """
    h, w = tile.shape[:2]
    bh, bw = max(2, int(RING * h)), max(2, int(RING * w))
    lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = np.ones((h, w), bool)
    ring[bh : h - bh, bw : w - bw] = False
    _, _, centers = cv2.kmeans(
        lab[ring], 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS
    )
    dist = np.min(np.linalg.norm(lab[..., None, :] - centers[None, None], axis=-1), axis=-1)
    clean = dist < 14

    ph, pw = bh // 2, bw // 2
    patches = []
    for y in range(0, h - ph + 1, ph):
        for x in range(0, w - pw + 1, pw):
            inside_ring = ring[y : y + ph, x : x + pw].all()
            if inside_ring and clean[y : y + ph, x : x + pw].all():
                patches.append(tile[y : y + ph, x : x + pw])
    if len(patches) < 12:
        return None
    out = np.zeros_like(tile)
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            patch = patches[rng.integers(len(patches))]
            out[y : y + ph, x : x + pw] = patch[: h - y, : w - x]
    return cv2.GaussianBlur(out, (0, 0), 1.0)


def harvest_backgrounds(images: list[np.ndarray], seed: int = 0) -> int:
    """Save a sprite-free background for every located tile (unlabeled screenshots are fine)."""
    from .grid import locate

    rng = np.random.default_rng(seed)
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in images:
        for grid in locate(img):
            for x0, y0, x1, y1 in grid.tiles():
                inset = 0.04 * (x1 - x0)  # clear of the border rim
                xa, ya, xb, yb = (int(round(v)) for v in (x0 + inset, y0 + inset, x1 - inset, y1 - inset))
                if xa < 0 or ya < 0 or xb > img.shape[1] or yb > img.shape[0] or xb - xa < 24:
                    continue  # tiles cut off by the crop
                tile = cv2.resize(img[ya:yb, xa:xb], (TILE, int(TILE * (yb - ya) / (xb - xa))))
                bg = sprite_free_background(tile, rng)
                if bg is None:
                    continue
                cv2.imwrite(str(BACKGROUNDS_DIR / f"{n:05d}.png"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
                n += 1
    return n


def load_backgrounds() -> list[np.ndarray]:
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(BACKGROUNDS_DIR.glob("*.png"))]


def random_spec(
    rng: np.random.Generator,
    palette,
    backgrounds=(),
    aspects=(0.94, 0.93),
    shadow: bool | None = None,
) -> TileSpec:
    state = rng.choice(["bright", "dim", "gray"], p=[0.45, 0.45, 0.10])
    fill = tuple(int(c) for c in palette[rng.integers(len(palette))])
    if shadow is None:
        shadow = bool(rng.random() < 0.3)
    background = None
    if len(backgrounds) and rng.random() < 0.5:
        background = backgrounds[rng.integers(len(backgrounds))]
        aspect = background.shape[0] / background.shape[1]
    else:
        aspect = float(aspects[rng.integers(len(aspects))])
    return TileSpec(state=str(state), shadow=shadow, fill=fill, aspect=aspect, background=background)


def synth_tiles(
    sprites: list[np.ndarray], n: int, rng: np.random.Generator, palette=None, backgrounds=()
) -> list[np.ndarray]:
    """`n` degraded synthetic tiles cycling through a class's reference images."""
    palette = palette or load_palette()
    out = []
    for i in range(n):
        sprite = sprites[i % len(sprites)]
        out.append(degrade(render_tile(sprite, random_spec(rng, palette, backgrounds), rng), rng))
    return out
