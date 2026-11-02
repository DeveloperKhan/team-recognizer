"""Locate team grids: the 2 columns x 3 rows of sprite tiles under each player's name bar.

Screenshots are cropped and scaled by hand, so nothing here assumes where the grid is or how big
it is. What is fixed is each overlay's *shape*: tile aspect ratio and gutter widths relative to the
tile (STYLES, measured from one clean frame per overlay). The image is searched, across scales,
for that 2x3 lattice of ten straight tile borders (4 vertical, 6 horizontal).

A border segment is scored by *coherent contrast*: the difference between the mean color of a
thin strip just inside it and a thin strip just outside it, damped by how non-uniform those strips
are. Tile fills and gutters are flat colors, so even the faint borders of dimmed tiles score;
camera boxes and game screens have a strong edge but texture on one side, and texture alone
averages out.

Search, coarse to fine:
  1. For each candidate tile width, resize so a tile is TILE_PX wide and score every lattice
     position for each style (integral images make every strip statistic O(1)).
  2. Keep the best lattices across scales, non-max suppressed.
  3. Re-measure at higher resolution and move each border line to its strongest position nearby.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Style:
    name: str
    aspect: float  # tile height / tile width
    gap_x: float  # vertical gutter / tile width
    gap_y: float  # horizontal gutter / tile width


# The 2027 Play! Pokémon GO broadcast overlay. Captures of it differ slightly in gutter width
# (stream scaling, highlighted-tile rims), so it is searched with two gutter settings; line
# snapping takes care of the rest.
STYLES = (
    Style("2027", aspect=0.94, gap_x=0.043, gap_y=0.046),  # e.g. Gdańsk full frames
    Style("2027-wide-gutter", aspect=0.93, gap_x=0.085, gap_y=0.08),  # e.g. Nice panel crops
)

TILE_PX = 32  # tile width during the coarse search
REFINE_PX = 96  # tile width during refinement (never upscaled)
SCALE_STEP = 1.06
MIN_TILE_PX = 16  # smallest tile, in original pixels, worth searching for
MIN_TILE_FRAC = 0.04  # ...and relative to the image's longer side
EDGE_SPAN = (0.2, 0.8)  # portion of each border scored, skipping rounded corners
STRIP = 0.08  # width of the strips compared on either side of a border, relative to tile width
CONTRAST_CAP = 60.0  # Lab units; one very strong border can't carry a lattice on its own
TEXTURE_SCALE = 6.0  # strip std (Lab units) at which a border's contrast counts half
MIN_SCORE = 2.0  # local maxima below this aren't kept as candidates
MIN_GRID_SCORE = 4.2  # the best lattice must reach this (with size prior) to count as a grid
SIZE_PRIOR = 0.2  # exponent on tile width / image size
SECOND_GRID_RATIO = 0.5  # the weaker team of a pair must score at least this fraction of the other
PAIR_CANDIDATES = 400


@dataclass
class Grid:
    """A located 2x3 team grid, in original image pixels."""

    xs: list[float]  # left0, right0, left1, right1
    ys: list[float]  # top0, bottom0, top1, bottom1, top2, bottom2
    score: float
    style: str = ""
    side: str = ""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.xs[0], self.ys[0], self.xs[3], self.ys[5]

    @property
    def tile_w(self) -> float:
        return ((self.xs[1] - self.xs[0]) + (self.xs[3] - self.xs[2])) / 2

    @property
    def tile_h(self) -> float:
        return ((self.ys[1] - self.ys[0]) + (self.ys[3] - self.ys[2]) + (self.ys[5] - self.ys[4])) / 3

    def tiles(self) -> list[tuple[float, float, float, float]]:
        """Six (x0, y0, x1, y1) boxes, row-major: top-left, top-right, middle-left, ..."""
        return [
            (self.xs[2 * c], self.ys[2 * r], self.xs[2 * c + 1], self.ys[2 * r + 1])
            for r in range(3)
            for c in range(2)
        ]

    def name_bar(self) -> tuple[float, float, float, float]:
        """Where the player's name bar sits: directly above the grid in every overlay seen."""
        x0, y0, x1, _ = self.bbox
        pad = 0.08 * (x1 - x0)
        return x0 - pad, y0 - 0.6 * self.tile_h, x1 + pad, y0 - 0.02 * self.tile_h

    def to_dict(self) -> dict:
        return {
            "bbox": [round(v, 1) for v in self.bbox],
            "score": round(self.score, 2),
            "style": self.style,
            "side": self.side,
        }


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def _lattice(w: float, style: Style) -> tuple[list[float], list[float]]:
    h, gx, gy = w * style.aspect, w * style.gap_x, w * style.gap_y
    xs = [0, w, w + gx, 2 * w + gx]
    ys = [0, h, h + gy, 2 * h + gy, 2 * h + 2 * gy, 3 * h + 2 * gy]
    return xs, ys


class _StripStats:
    """Mean and texture (std) of every bh x bw box, indexed by top-left corner."""

    def __init__(self, s: np.ndarray, sq: np.ndarray, bh: int, bw: int):
        n = bh * bw
        box = lambda I: I[bh:, bw:] - I[:-bh, bw:] - I[bh:, :-bw] + I[:-bh, :-bw]  # noqa: E731
        self.mean = box(s) / n
        var = (box(sq) / n - self.mean**2).clip(min=0).sum(-1)
        self.std = np.sqrt(var / 3)


def _segment(a_mean, a_std, b_mean, b_std) -> np.ndarray:
    contrast = np.minimum(np.linalg.norm(a_mean - b_mean, axis=-1), CONTRAST_CAP)
    texture = np.maximum(a_std, b_std)
    return contrast / (1 + texture / TEXTURE_SCALE)


def _aggregate(segments: list[np.ndarray]) -> np.ndarray:
    """Mean border score, blended with a low quantile so most borders have to be present."""
    stack = np.stack(segments)
    k = len(segments) // 4
    low = np.partition(stack, k, axis=0)[k]
    return 0.5 * stack.mean(0) + 0.5 * low


def _score_map(s: np.ndarray, sq: np.ndarray, w: float, style: Style) -> np.ndarray:
    """Lattice score for every top-left position (rows = y, cols = x), offset by the strip width."""
    xs, ys = (list(map(lambda v: int(round(v)), vs)) for vs in _lattice(w, style))
    h = w * style.aspect
    a = max(1, int(round(STRIP * w)))
    lv = max(2, int(round((EDGE_SPAN[1] - EDGE_SPAN[0]) * h)))
    lh = max(2, int(round((EDGE_SPAN[1] - EDGE_SPAN[0]) * w)))
    vert = _StripStats(s, sq, lv, a)
    horiz = _StripStats(s, sq, a, lh)
    # Borders facing a gutter compare against the gutter itself, not the next tile over.
    gx = max(1, min(a, int(round(style.gap_x * w))))
    gy = max(1, min(a, int(round(style.gap_y * w))))
    vert_gutter = _StripStats(s, sq, lv, gx)
    horiz_gutter = _StripStats(s, sq, gy, lh)
    H, W = s.shape[0] - 1, s.shape[1] - 1
    out_h, out_w = H - ys[-1] - 2 * a - 1, W - xs[-1] - 2 * a - 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0), np.float32)

    def at(stats: _StripStats, y: int, x: int):
        return stats.mean[y : y + out_h, x : x + out_w], stats.std[y : y + out_h, x : x + out_w]

    segments = []
    for r in range(3):
        y = a + ys[2 * r] + int(round(EDGE_SPAN[0] * h))
        for i, x in enumerate(xs):
            x += a
            if i == 1:  # right edge of the left column: tile on the left, gutter on the right
                segments.append(_segment(*at(vert, y, x - a), *at(vert_gutter, y, x)))
            elif i == 2:  # left edge of the right column: gutter on the left
                segments.append(_segment(*at(vert_gutter, y, x - gx), *at(vert, y, x)))
            else:
                segments.append(_segment(*at(vert, y, x - a), *at(vert, y, x)))
    for c in range(2):
        x = a + xs[2 * c] + int(round(EDGE_SPAN[0] * w))
        for i, y in enumerate(ys):
            y += a
            if i in (1, 3):  # bottom of a row with a gutter below
                segments.append(_segment(*at(horiz, y - a, x), *at(horiz_gutter, y, x)))
            elif i in (2, 4):  # top of a row with a gutter above
                segments.append(_segment(*at(horiz_gutter, y - gy, x), *at(horiz, y, x)))
            else:
                segments.append(_segment(*at(horiz, y - a, x), *at(horiz, y, x)))
    return _aggregate(segments)


def _local_maxima(score: np.ndarray, k: int) -> list[tuple[float, int, int]]:
    if score.size == 0:
        return []
    dil = cv2.dilate(score, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((score >= dil) & (score > MIN_SCORE))
    order = np.argsort(-score[ys, xs])[:k]
    return [(float(score[ys[i], xs[i]]), int(ys[i]), int(xs[i])) for i in order]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _coarse(img: np.ndarray, styles: tuple[Style, ...]) -> list[Grid]:
    H, W = img.shape[:2]
    candidates: list[Grid] = []
    tile = max(MIN_TILE_PX, MIN_TILE_FRAC * max(H, W))
    # The grid (2 tiles wide, 3 tall) may run a little off the crop, but not much.
    max_tile = min(W / 1.8, H / (2.7 * min(s.aspect for s in styles)))
    while tile <= max_tile:
        r = TILE_PX / tile
        small = cv2.resize(img, (max(1, round(W * r)), max(1, round(H * r))), interpolation=cv2.INTER_AREA)
        pad = TILE_PX // 3
        small = _lab(cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
        s, sq = cv2.integral2(small, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
        for style in styles:
            a = max(1, int(round(STRIP * TILE_PX)))
            # Small lattices turn up by chance inside sprites and footage far more often than
            # large ones, so the score is weighted toward grids that fill more of the image.
            size_prior = (tile / max(H, W)) ** SIZE_PRIOR
            for score, y, x in _local_maxima(_score_map(s, sq, TILE_PX, style), k=6):
                xs, ys = _lattice(TILE_PX, style)
                candidates.append(
                    Grid(
                        xs=[(x + a + v - pad) / r for v in xs],
                        ys=[(y + a + v - pad) / r for v in ys],
                        score=score * size_prior,
                        style=style.name,
                    )
                )
        tile *= SCALE_STEP
    candidates.sort(key=lambda g: -g.score)
    return candidates


def _snap(img: np.ndarray, grid: Grid) -> Grid:
    """Move each border line to the position of strongest coherent contrast near it."""
    r = min(1.0, REFINE_PX / grid.tile_w)
    H, W = img.shape[:2]
    lab = _lab(cv2.resize(img, (max(1, round(W * r)), max(1, round(H * r))), interpolation=cv2.INTER_AREA))
    sh, sw = lab.shape[:2]
    xs = [v * r for v in grid.xs]
    ys = [v * r for v in grid.ys]
    tw, th = grid.tile_w * r, grid.tile_h * r
    a = max(1, int(round(STRIP * tw)))
    radius = max(2, int(round(0.05 * tw)))

    def score(p: np.ndarray, q: np.ndarray) -> float:
        contrast = min(float(np.linalg.norm(p.mean((0, 1)) - q.mean((0, 1)))), CONTRAST_CAP)
        texture = max(p.std((0, 1)).mean(), q.std((0, 1)).mean())
        return contrast / (1 + texture / TEXTURE_SCALE)

    def v_score(x: int) -> float:
        vals = []
        for row in range(3):
            y0 = int(max(0, ys[2 * row] + EDGE_SPAN[0] * th))
            y1 = int(min(sh, ys[2 * row] + EDGE_SPAN[1] * th))
            if y1 - y0 > 2 and x - a >= 0 and x + a <= sw:
                vals.append(score(lab[y0:y1, x - a : x], lab[y0:y1, x : x + a]))
        return float(np.mean(vals)) if vals else -1.0

    def h_score(y: int) -> float:
        vals = []
        for col in range(2):
            x0 = int(max(0, xs[2 * col] + EDGE_SPAN[0] * tw))
            x1 = int(min(sw, xs[2 * col] + EDGE_SPAN[1] * tw))
            if x1 - x0 > 2 and y - a >= 0 and y + a <= sh:
                vals.append(score(lab[y - a : y, x0:x1], lab[y : y + a, x0:x1]))
        return float(np.mean(vals)) if vals else -1.0

    def best(fn, pos: float) -> float:
        p0 = int(round(pos))
        val, p = max((fn(p), p) for p in range(p0 - radius, p0 + radius + 1))
        return float(p) if val > 0 else pos

    return Grid(
        xs=[best(v_score, x) / r for x in xs],
        ys=[best(h_score, y) / r for y in ys],
        score=grid.score,
        style=grid.style,
    )


def _overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _regularize(grid: Grid) -> Grid:
    """Least-squares fit of a regular lattice (equal tiles, equal gutters) to snapped lines."""

    def fit(values, pattern):
        # values[i] = origin + size * pattern[i][0] + gap * pattern[i][1]
        A = np.array([[1, a, b] for a, b in pattern], np.float64)
        coef, *_ = np.linalg.lstsq(A, np.array(values, np.float64), rcond=None)
        return [float(v) for v in A @ coef]

    xs = fit(grid.xs, [(0, 0), (1, 0), (1, 1), (2, 1)])
    ys = fit(grid.ys, [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)])
    return Grid(xs=xs, ys=ys, score=grid.score, style=grid.style)


def _refine(img: np.ndarray, grid: Grid, rounds: int = 3) -> Grid:
    """Snap lines, refit a regular lattice, repeat; then a final free snap. A coarse lattice
    with a slightly wrong aspect ratio is off by more on each row down; the refit carries each
    round's corrections of the upper rows down to the lower ones."""
    for _ in range(rounds):
        grid = _regularize(_snap(img, grid))
    return _snap(img, grid)


def _partners(a: Grid, b: Grid) -> bool:
    """Whether two lattices could be the two teams of one broadcast: same overlay, same scale,
    side by side at the same height."""
    return (
        a.style == b.style
        and abs(a.tile_w - b.tile_w) <= 0.08 * a.tile_w
        and abs(a.ys[0] - b.ys[0]) <= 0.15 * a.tile_h
        # The two teams flank the game screens; adjacent lattices are one grid split in two.
        and max(a.bbox[0], b.bbox[0]) - min(a.bbox[2], b.bbox[2]) >= 2 * a.tile_w
    )


def locate(img: np.ndarray, max_grids: int = 2, styles: tuple[Style, ...] = STYLES) -> list[Grid]:
    """Find the team grids in an RGB image (one for a single panel, two for a full broadcast frame),
    left to right.

    A full frame shows both teams at the same scale and height, so pairs are scored together: a
    matched pair outranks any single lattice that has no partner, which is what keeps game
    screens and camera boxes from winning in full frames.
    """
    candidates = _coarse(img, styles)[:PAIR_CANDIDATES]
    if not candidates:
        return []
    best: list[Grid] = [candidates[0]]
    # A pair has to clearly beat the best single grid, or the crop shows one team.
    best_total = candidates[0].score * (1 + SECOND_GRID_RATIO)
    if max_grids >= 2:
        for i, a in enumerate(candidates):
            if a.score * 2 <= best_total:
                break
            for b in candidates[i + 1 :]:
                if a.score + b.score < best_total:
                    break
                if b.score >= a.score * SECOND_GRID_RATIO and _partners(a, b):
                    best, best_total = [a, b], a.score + b.score
                    break
    if best[0].score < MIN_GRID_SCORE:
        return []
    grids = sorted((_refine(img, g) for g in best), key=lambda g: g.bbox[0])
    if len(grids) == 2:
        grids[0].side, grids[1].side = "left", "right"
    return grids
