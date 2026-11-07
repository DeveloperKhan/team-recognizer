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

from .paths import CACHE, SMOKE

TILE = 160  # synthetic tile width; height follows the style's aspect ratio
PALETTE_FILE = CACHE / "palette.json"
BACKGROUNDS_DIR = CACHE / "backgrounds"
RING = 0.16  # border band (fraction of the tile) searched for sprite-free patches
# The overlay draws a Pokémon nearly as large as its tile: its larger side spans this fraction of
# the tile's, measured on labeled broadcast tiles. Centre jitter is a fraction of the tile.
SPRITE_FIT = (0.84, 1.05)
SPRITE_SHIFT = ((-0.03, 0.03), (-0.08, 0.04))

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
    opacity: float | None = None  # a dimmed sprite's opacity (random when None)
    smoke: float | None = None  # shadow smoke strength (random when None)


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
    ox0, oy0, ox1, oy1 = (
        clip(x0 + outer * w, W),
        clip(y0 + outer * h, H),
        clip(x1 - outer * w, W),
        clip(y1 - outer * h, H),
    )
    ix0, iy0, ix1, iy1 = (
        clip(x0 + inner * w, W),
        clip(y0 + inner * h, H),
        clip(x1 - inner * w, W),
        clip(y1 - inner * h, H),
    )
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


def extract_smoke(screenshot: np.ndarray) -> np.ndarray:
    """A shadow smoke texture (RGBA) from a screenshot of one shadow tile: each pixel taken as a
    mix of the tile's green and the smoke's purple, its opacity how far along that line it sits.
    Pixels well off the line (the Pokémon) get none, so only the smoke is kept."""
    hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV).astype(int)
    green = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 40)
    # The largest green region is the tile; trim its rounded border
    _, _, stats, _ = cv2.connectedComponentsWithStats(green.astype(np.uint8))
    x, y, w, h = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), :4]
    inset = int(0.045 * max(w, h))
    tile = screenshot[y + inset : y + h - inset, x + inset : x + w - inset]

    t = tile.astype(np.float32)
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV).astype(int)
    strong = (hsv[..., 0] >= 116) & (hsv[..., 0] <= 165) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 50)
    green = (hsv[..., 0] >= 40) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50)
    if strong.sum() < 20 or green.sum() < 20:
        return np.zeros(tile.shape[:2] + (4,), np.uint8)  # no smoke to speak of (dimmed, fainted)
    purple = np.median(t[strong], 0)
    # The green varies across the tile: average it locally, from green pixels only
    weight = cv2.blur(green.astype(np.float32), (41, 41))[..., None]
    bg = cv2.blur(np.where(green[..., None], t, 0), (41, 41)) / np.maximum(weight, 1e-3)
    bg = np.where(weight > 0.05, bg, np.median(t[green], 0))
    d = purple - bg
    along = ((t - bg) * d).sum(-1) / np.maximum((d * d).sum(-1), 1)
    off = np.linalg.norm(t - bg - along[..., None] * d, axis=-1)
    alpha = np.clip((along - 0.12) / 0.75, 0, 1) * (off < 28)
    alpha[cv2.dilate((off > 40).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    rgba = np.dstack([np.broadcast_to(purple, t.shape[:2] + (3,)), alpha * 255]).astype(np.uint8)
    dense = alpha > 0.6  # the real smoke's shading where it's dense, its median purple elsewhere
    rgba[dense, :3] = tile[dense]
    return rgba


_smoke_textures: list[np.ndarray] | None = None


def load_smoke() -> list[np.ndarray]:
    """The smoke textures in the data folder (paths.SMOKE), RGBA; written by `recognizer smoke`."""
    global _smoke_textures
    if _smoke_textures is None:
        files = sorted(SMOKE.glob("*.png")) if SMOKE.is_dir() else []
        _smoke_textures = [np.asarray(Image.open(f).convert("RGBA")) for f in files]
    return _smoke_textures


def real_smoke(h: int, w: int, rng: np.random.Generator, textures: list[np.ndarray]):
    """Smoke for an h x w tile from the real textures: one picked, flipped and turned (the smoke
    hugs the border, so it's never cropped), its color nudged, often with more layered over it.
    Returns (rgb, alpha) as floats."""

    def one():
        tex = textures[rng.integers(len(textures))]
        if rng.random() < 0.5:
            tex = tex[:, ::-1]
        if rng.random() < 0.5:
            tex = tex[::-1]
        tex = np.ascontiguousarray(np.rot90(tex, int(rng.integers(4))))
        tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        return tex[..., :3] * rng.uniform(0.9, 1.1, 3), tex[..., 3:4] / 255

    rgb, alpha = one()
    for _ in range(int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))):
        rgb2, alpha2 = one()
        rgb = np.where(alpha2 > alpha, rgb2, rgb)
        alpha = np.maximum(alpha, alpha2)
    return np.clip(rgb, 0, 255), alpha


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

    if spec.shadow:
        textures = load_smoke()
        strength = spec.smoke or (rng.uniform(0.85, 1.0) if spec.state == "bright" else rng.uniform(0.35, 0.7))
        if textures:
            # Smoke cut from real shadow tiles, on flat fills and real backgrounds alike
            purple, m = real_smoke(h, w, rng, textures)
            tile = tile * (1 - m * strength) + purple * m * strength
        elif spec.background is None:
            m = smoke_mask(h, w, rng)[..., None]
            purple = np.array(SMOKE_PURPLE[rng.integers(len(SMOKE_PURPLE))], np.float32)
            strength = rng.uniform(0.75, 1.0) if spec.state == "bright" else rng.uniform(0.2, 0.55)
            tile = tile * (1 - m * strength) + purple * m * strength

    # Scale the sprite to fit, with the jitter seen between overlays and animation frames.
    sh, sw = sprite.shape[:2]
    fit = rng.uniform(*SPRITE_FIT) * min(w / sw, h / sh)
    nw, nh = max(1, int(sw * fit)), max(1, int(sh * fit))
    spr = np.asarray(Image.fromarray(sprite).resize((nw, nh), Image.LANCZOS)).astype(np.float32)
    (sx0, sx1), (sy0, sy1) = SPRITE_SHIFT
    x0 = int((w - nw) / 2 + rng.uniform(sx0, sx1) * w)
    y0 = int((h - nh) / 2 + rng.uniform(sy0, sy1) * h)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x0 + nw), min(h, y0 + nh)
    crop = spr[ya - y0 : yb - y0, xa - x0 : xb - x0]
    alpha = crop[..., 3:4] / 255
    rgb = crop[..., :3]
    if spec.state == "dim":
        # Dimmed sprites are see-through (the tile, smoke included, shows behind them) and
        # slightly tinted toward the fill.
        alpha = alpha * (spec.opacity or rng.uniform(0.3, 0.65))
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

    The tile's fill is a smooth gradient, so a low-order surface fitted to the clean fill pixels
    of its border band (green, not smoke, not sprite) reproduces it over the whole tile: its
    color, gradient and darker corners, with no seams. Smoke isn't kept; shadow tiles get theirs
    from the real smoke textures.
    """
    h, w = tile.shape[:2]
    bh, bw = max(2, int(RING * h)), max(2, int(RING * w))
    ring = np.ones((h, w), bool)
    ring[bh : h - bh, bw : w - bw] = False
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV).astype(int)
    green = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 30)
    candidates = ring & green
    if candidates.sum() < 200:
        return None
    # Sprites reach into the band (wings, tails): keep pixels close to the band's typical fill
    lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB).astype(np.float32)
    fill = np.median(lab[candidates], 0)
    clean = candidates & (np.linalg.norm(lab - fill, axis=-1) < 22)
    if clean.sum() < 200:
        return None

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / (w - 1) * 2 - 1, yy / (h - 1) * 2 - 1
    basis = np.stack([np.ones_like(u), u, v, u * u, v * v, u * v], -1)
    coef, *_ = np.linalg.lstsq(basis[clean], tile[clean].astype(np.float32), rcond=None)
    out = basis @ coef
    out += rng.normal(0, 1.5, out.shape)  # the fill's faint grain
    return np.clip(cv2.GaussianBlur(out, (0, 0), 0.6), 0, 255).astype(np.uint8)


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


# The plain look in every state the overlay shows it, at set opacities and smoke strengths:
# (state, opacity when dimmed, shadow, smoke strength)
PLAIN_GRID = [
    ("bright", None, False, None),
    ("bright", None, False, None),
    ("dim", 0.3, False, None),
    ("dim", 0.45, False, None),
    ("dim", 0.6, False, None),
    ("gray", None, False, None),
    ("bright", None, True, 0.6),
    ("bright", None, True, 0.95),
    ("dim", 0.35, True, 0.45),
    ("dim", 0.55, True, 0.65),
    ("dim", 0.45, True, 0.55),
    ("gray", None, True, 0.8),
]


def plain_variants(sprite: np.ndarray, rng: np.random.Generator, palette=None, backgrounds=()) -> list[np.ndarray]:
    """Degraded synthetic tiles of one image (a class's plain art) across PLAIN_GRID."""
    palette = palette or load_palette()
    out = []
    for state, opacity, shadow, smoke in PLAIN_GRID:
        spec = random_spec(rng, palette, backgrounds, shadow=shadow)
        spec.state, spec.opacity, spec.smoke = state, opacity, smoke
        out.append(degrade(render_tile(sprite, spec, rng), rng))
    return out


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
