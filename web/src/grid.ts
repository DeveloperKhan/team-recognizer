/**
 * Locate team grids: port of recognizer/grid.py (see its docstring for the method).
 *
 * The 2x3 lattice of tile borders is searched across scales; each border segment is scored by
 * coherent contrast between thin strips on either side of it, damped by how textured those
 * strips are. Integral images make every strip statistic O(1). Structure, constants and
 * rounding follow the Python code so both find the same grids.
 */

import type { GridConfig, Style } from "./config.ts";
import { AreaSampler, padReplicate, pyRound, resizeArea, rgbToLab, type RGBImage } from "./image.ts";

export class Grid {
  xs: number[]; // left0, right0, left1, right1
  ys: number[]; // top0, bottom0, top1, bottom1, top2, bottom2
  score: number;
  style: string;
  side: "left" | "right" | null = null;

  constructor(xs: number[], ys: number[], score: number, style: string) {
    this.xs = xs;
    this.ys = ys;
    this.score = score;
    this.style = style;
  }

  get bbox(): [number, number, number, number] {
    return [this.xs[0], this.ys[0], this.xs[3], this.ys[5]];
  }

  get tileW(): number {
    return (this.xs[1] - this.xs[0] + (this.xs[3] - this.xs[2])) / 2;
  }

  get tileH(): number {
    return (this.ys[1] - this.ys[0] + (this.ys[3] - this.ys[2]) + (this.ys[5] - this.ys[4])) / 3;
  }

  /** Six [x0, y0, x1, y1] boxes, row-major. */
  tiles(): Array<[number, number, number, number]> {
    const out: Array<[number, number, number, number]> = [];
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 2; c++) out.push([this.xs[2 * c], this.ys[2 * r], this.xs[2 * c + 1], this.ys[2 * r + 1]]);
    return out;
  }

  nameBar(): [number, number, number, number] {
    const [x0, y0, x1] = this.bbox;
    const pad = 0.08 * (x1 - x0);
    return [x0 - pad, y0 - 0.6 * this.tileH, x1 + pad, y0 - 0.02 * this.tileH];
  }

  countryTag(): [number, number, number, number] {
    const [x0, y0, x1] = this.bbox;
    const pad = 0.08 * (x1 - x0);
    return [x0 - pad, y0 - 0.95 * this.tileH, x1 + pad, y0 - 0.45 * this.tileH];
  }
}

function lattice(w: number, style: Style): [number[], number[]] {
  const h = w * style.aspect, gx = w * style.gapX, gy = w * style.gapY;
  return [
    [0, w, w + gx, 2 * w + gx],
    [0, h, h + gy, 2 * h + gy, 2 * h + 2 * gy, 3 * h + 2 * gy],
  ];
}

/** Integral images of a 3-channel float image and of its squares ((H+1) x (W+1) x 3). */
function integrals(lab: Float32Array, width: number, height: number) {
  const W1 = width + 1;
  const s = new Float64Array((height + 1) * W1 * 3);
  const sq = new Float64Array((height + 1) * W1 * 3);
  for (let y = 0; y < height; y++) {
    const row = [0, 0, 0];
    const rowSq = [0, 0, 0];
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      const o = ((y + 1) * W1 + x + 1) * 3;
      const p = (y * W1 + x + 1) * 3;
      for (let c = 0; c < 3; c++) {
        const v = lab[i + c];
        row[c] += v;
        rowSq[c] += v * v;
        s[o + c] = s[p + c] + row[c];
        sq[o + c] = sq[p + c] + rowSq[c];
      }
    }
  }
  return { s, sq, width, height };
}

type Integrals = ReturnType<typeof integrals>;

/** Mean (3 channels) and texture of every bh x bw box, indexed by top-left corner. */
class StripStats {
  readonly cols: number;
  readonly mean: Float32Array;
  readonly std: Float32Array;

  constructor(I: Integrals, bh: number, bw: number) {
    const W1 = I.width + 1;
    const rows = I.height + 1 - bh;
    this.cols = W1 - bw;
    const n = bh * bw;
    this.mean = new Float32Array(rows * this.cols * 3);
    this.std = new Float32Array(rows * this.cols);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const a = (y * W1 + x) * 3, b = (y * W1 + x + bw) * 3;
        const c = ((y + bh) * W1 + x) * 3, d = ((y + bh) * W1 + x + bw) * 3;
        let variance = 0;
        const o = y * this.cols + x;
        for (let ch = 0; ch < 3; ch++) {
          const m = (I.s[d + ch] - I.s[b + ch] - I.s[c + ch] + I.s[a + ch]) / n;
          const m2 = (I.sq[d + ch] - I.sq[b + ch] - I.sq[c + ch] + I.sq[a + ch]) / n;
          this.mean[o * 3 + ch] = m;
          variance += Math.max(0, m2 - m * m);
        }
        this.std[o] = Math.sqrt(variance / 3);
      }
    }
  }
}

function scoreMap(I: Integrals, w: number, style: Style, cfg: GridConfig) {
  const [fxs, fys] = lattice(w, style);
  const xs = fxs.map(pyRound);
  const ys = fys.map(pyRound);
  const h = w * style.aspect;
  const [e0, e1] = cfg.edgeSpan;
  const a = Math.max(1, pyRound(cfg.strip * w));
  const lv = Math.max(2, pyRound((e1 - e0) * h));
  const lh = Math.max(2, pyRound((e1 - e0) * w));
  const gx = Math.max(1, Math.min(a, pyRound(style.gapX * w)));
  const gy = Math.max(1, Math.min(a, pyRound(style.gapY * w)));
  const outH = I.height - ys[5] - 2 * a - 1;
  const outW = I.width - xs[3] - 2 * a - 1;
  if (outH <= 0 || outW <= 0) return null;

  const vert = new StripStats(I, lv, a);
  const horiz = new StripStats(I, a, lh);
  const vertGutter = new StripStats(I, lv, gx);
  const horizGutter = new StripStats(I, gy, lh);

  // Each segment compares strip P (at py, px) with strip Q (at qy, qx) for every position.
  type Seg = [StripStats, number, number, StripStats, number, number];
  const segs: Seg[] = [];
  for (let r = 0; r < 3; r++) {
    const y = a + ys[2 * r] + pyRound(e0 * h);
    xs.forEach((x0, i) => {
      const x = x0 + a;
      if (i === 1) segs.push([vert, y, x - a, vertGutter, y, x]);
      else if (i === 2) segs.push([vertGutter, y, x - gx, vert, y, x]);
      else segs.push([vert, y, x - a, vert, y, x]);
    });
  }
  for (let c = 0; c < 2; c++) {
    const x = a + xs[2 * c] + pyRound(e0 * w);
    ys.forEach((y0, i) => {
      const y = y0 + a;
      if (i === 1 || i === 3) segs.push([horiz, y - a, x, horizGutter, y, x]);
      else if (i === 2 || i === 4) segs.push([horizGutter, y - gy, x, horiz, y, x]);
      else segs.push([horiz, y - a, x, horiz, y, x]);
    });
  }

  // Score every segment for every position, stored position-major so the quantile below reads
  // each position's values contiguously.
  const n = segs.length;
  const k = Math.floor(n / 4);
  const positions = outH * outW;
  const all = new Float32Array(positions * n);
  segs.forEach(([P, y1, x1, Q, y2, x2], s) => {
    const pm = P.mean, qm = Q.mean, ps = P.std, qs = Q.std;
    const invTexture = 1 / cfg.textureScale;
    for (let py = 0; py < outH; py++) {
      let i = (y1 + py) * P.cols + x1;
      let j = (y2 + py) * Q.cols + x2;
      let o = py * outW * n + s;
      for (let px = 0; px < outW; px++, i++, j++, o += n) {
        const dr = pm[i * 3] - qm[j * 3];
        const dg = pm[i * 3 + 1] - qm[j * 3 + 1];
        const db = pm[i * 3 + 2] - qm[j * 3 + 2];
        const contrast = Math.min(Math.sqrt(dr * dr + dg * dg + db * db), cfg.contrastCap);
        const texture = ps[i] > qs[j] ? ps[i] : qs[j];
        all[o] = contrast / (1 + texture * invTexture);
      }
    }
  });

  // Blend the mean with a low quantile (np.partition(stack, k)[k]) so most borders must be present.
  const out = new Float32Array(positions);
  const buf = new Float64Array(n);
  for (let p = 0, o = 0; p < positions; p++) {
    let sum = 0;
    for (let s = 0; s < n; s++, o++) {
      buf[s] = all[o];
      sum += all[o];
    }
    out[p] = 0.5 * (sum / n) + 0.5 * kthSmallest(buf, k);
  }
  return { out, outH, outW, a };
}

/** The k-th smallest value (0-based) of a small array; reorders it (quickselect). */
function kthSmallest(a: Float64Array, k: number): number {
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i]; a[i] = a[j]; a[j] = t;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/** Top-k local maxima (5x5 neighbourhood) above minScore, as [score, y, x]. */
function localMaxima(score: Float32Array, H: number, W: number, k: number, minScore: number) {
  const found: Array<[number, number, number]> = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = score[y * W + x];
      if (v <= minScore) continue;
      let isMax = true;
      for (let dy = -2; dy <= 2 && isMax; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx;
          if (xx >= 0 && xx < W && score[yy * W + xx] > v) {
            isMax = false;
            break;
          }
        }
      }
      if (isMax) found.push([v, y, x]);
    }
  }
  return found.sort((p, q) => q[0] - p[0]).slice(0, k);
}

function coarse(img: RGBImage, cfg: GridConfig): Grid[] {
  const { width: W, height: H } = img;
  const candidates: Grid[] = [];
  let tile = Math.max(cfg.minTilePx, cfg.scaleAnchorFrac * Math.max(H, W));
  while (tile < Math.max(cfg.minTilePx, cfg.minTileFrac * Math.max(H, W))) tile *= cfg.scaleStep;
  const maxTile = Math.min(W / 1.8, H / (2.7 * Math.min(...cfg.styles.map((s) => s.aspect))));
  const pad = Math.floor(cfg.tilePx / 3);
  const a = Math.max(1, pyRound(cfg.strip * cfg.tilePx));
  const sampler = new AreaSampler(img);
  while (tile <= maxTile) {
    const r = cfg.tilePx / tile;
    const small = sampler.resize(Math.max(1, pyRound(W * r)), Math.max(1, pyRound(H * r)));
    const padded = padReplicate(rgbToLab(small), small.width, small.height, 3, pad);
    const I = integrals(padded.data, padded.width, padded.height);
    // Small lattices turn up by chance inside sprites and footage far more often than large
    // ones, so the score is weighted toward grids that fill more of the image.
    const sizePrior = Math.pow(tile / Math.max(H, W), cfg.sizePrior);
    for (const style of cfg.styles) {
      const map = scoreMap(I, cfg.tilePx, style, cfg);
      if (!map) continue;
      const [xs, ys] = lattice(cfg.tilePx, style);
      for (const [score, y, x] of localMaxima(map.out, map.outH, map.outW, 6, cfg.minScore)) {
        candidates.push(
          new Grid(
            xs.map((v) => (x + a + v - pad) / r),
            ys.map((v) => (y + a + v - pad) / r),
            score * sizePrior,
            style.name
          )
        );
      }
    }
    tile *= cfg.scaleStep;
  }
  return candidates.sort((p, q) => q.score - p.score);
}

/** Mean and per-channel std of a rectangle of a Lab image. */
function regionStats(lab: Float32Array, width: number, x0: number, y0: number, x1: number, y1: number) {
  const n = (x1 - x0) * (y1 - y0);
  const sum = [0, 0, 0], sq = [0, 0, 0];
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 3;
      for (let c = 0; c < 3; c++) {
        sum[c] += lab[i + c];
        sq[c] += lab[i + c] * lab[i + c];
      }
    }
  }
  const mean = sum.map((v) => v / n);
  const std = sq.map((v, c) => Math.sqrt(Math.max(0, v / n - mean[c] * mean[c])));
  return { mean, texture: (std[0] + std[1] + std[2]) / 3 };
}

/** Move each border line to the position of strongest coherent contrast near it. */
function snap(img: RGBImage, grid: Grid, cfg: GridConfig): Grid {
  const r = Math.min(1, cfg.refinePx / grid.tileW);
  const small = resizeArea(img, Math.max(1, pyRound(img.width * r)), Math.max(1, pyRound(img.height * r)));
  const lab = rgbToLab(small);
  const sw = small.width, sh = small.height;
  const xs = grid.xs.map((v) => v * r);
  const ys = grid.ys.map((v) => v * r);
  const tw = grid.tileW * r, th = grid.tileH * r;
  const a = Math.max(1, pyRound(cfg.strip * tw));
  const radius = Math.max(2, pyRound(0.05 * tw));
  const [e0, e1] = cfg.edgeSpan;

  const score = (p: ReturnType<typeof regionStats>, q: ReturnType<typeof regionStats>) => {
    const d = Math.hypot(p.mean[0] - q.mean[0], p.mean[1] - q.mean[1], p.mean[2] - q.mean[2]);
    return Math.min(d, cfg.contrastCap) / (1 + Math.max(p.texture, q.texture) / cfg.textureScale);
  };
  const vScore = (x: number) => {
    const vals: number[] = [];
    for (let row = 0; row < 3; row++) {
      const y0 = Math.trunc(Math.max(0, ys[2 * row] + e0 * th));
      const y1 = Math.trunc(Math.min(sh, ys[2 * row] + e1 * th));
      if (y1 - y0 > 2 && x - a >= 0 && x + a <= sw)
        vals.push(score(regionStats(lab, sw, x - a, y0, x, y1), regionStats(lab, sw, x, y0, x + a, y1)));
    }
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : -1;
  };
  const hScore = (y: number) => {
    const vals: number[] = [];
    for (let col = 0; col < 2; col++) {
      const x0 = Math.trunc(Math.max(0, xs[2 * col] + e0 * tw));
      const x1 = Math.trunc(Math.min(sw, xs[2 * col] + e1 * tw));
      if (x1 - x0 > 2 && y - a >= 0 && y + a <= sh)
        vals.push(score(regionStats(lab, sw, x0, y - a, x1, y), regionStats(lab, sw, x0, y, x1, y + a)));
    }
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : -1;
  };
  const best = (fn: (p: number) => number, pos: number) => {
    const p0 = pyRound(pos);
    let bestVal = -Infinity, bestP = p0;
    for (let p = p0 - radius; p <= p0 + radius; p++) {
      const v = fn(p);
      if (v >= bestVal) {
        // Ties go to the larger offset, as Python's max over (value, position) tuples does.
        bestVal = v;
        bestP = p;
      }
    }
    return bestVal > 0 ? bestP : pos;
  };
  return new Grid(
    xs.map((x) => best(vScore, x) / r),
    ys.map((y) => best(hScore, y) / r),
    grid.score,
    grid.style
  );
}

/** Least-squares fit of a regular lattice (equal tiles, equal gutters) to snapped lines. */
function regularize(grid: Grid): Grid {
  const fit = (values: number[], pattern: Array<[number, number]>) => {
    // values[i] ≈ origin + size * p + gap * q; solve the 3x3 normal equations.
    const A = pattern.map(([p, q]) => [1, p, q]);
    const M = [0, 1, 2].map((i) => [0, 1, 2].map((j) => A.reduce((s, row) => s + row[i] * row[j], 0)));
    const v = [0, 1, 2].map((i) => A.reduce((s, row, n) => s + row[i] * values[n], 0));
    const coef = solve3(M, v);
    return A.map((row) => row[0] * coef[0] + row[1] * coef[1] + row[2] * coef[2]);
  };
  return new Grid(
    fit(grid.xs, [[0, 0], [1, 0], [1, 1], [2, 1]]),
    fit(grid.ys, [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2]]),
    grid.score,
    grid.style
  );
}

function solve3(M: number[][], v: number[]): number[] {
  const a = M.map((row, i) => [...row, v[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    for (let r = 0; r < 3; r++) {
      if (r === c) continue;
      const f = a[r][c] / a[c][c];
      for (let k = c; k < 4; k++) a[r][k] -= f * a[c][k];
    }
  }
  return [0, 1, 2].map((i) => a[i][3] / a[i][i]);
}

function refine(img: RGBImage, grid: Grid, cfg: GridConfig, rounds = 3): Grid {
  for (let i = 0; i < rounds; i++) grid = regularize(snap(img, grid, cfg));
  return snap(img, grid, cfg);
}

function partners(a: Grid, b: Grid): boolean {
  return (
    a.style === b.style &&
    Math.abs(a.tileW - b.tileW) <= 0.08 * a.tileW &&
    Math.abs(a.ys[0] - b.ys[0]) <= 0.15 * a.tileH &&
    // The two teams flank the game screens; adjacent lattices are one grid split in two.
    Math.max(a.bbox[0], b.bbox[0]) - Math.min(a.bbox[2], b.bbox[2]) >= 2 * a.tileW
  );
}

/** The team grids in an image (one for a single panel, two for a full broadcast frame), left to right. */
export function locate(img: RGBImage, cfg: GridConfig): Grid[] {
  const candidates = coarse(img, cfg).slice(0, cfg.pairCandidates);
  if (!candidates.length) return [];
  let best: Grid[] = [candidates[0]];
  // A pair has to clearly beat the best single grid, or the crop shows one team.
  let bestTotal = candidates[0].score * (1 + cfg.secondGridRatio);
  outer: for (let i = 0; i < candidates.length; i++) {
    const a = candidates[i];
    if (a.score * 2 <= bestTotal) break;
    for (let j = i + 1; j < candidates.length; j++) {
      const b = candidates[j];
      if (a.score + b.score < bestTotal) continue outer;
      if (b.score >= a.score * cfg.secondGridRatio && partners(a, b)) {
        best = [a, b];
        bestTotal = a.score + b.score;
        continue outer;
      }
    }
  }
  if (best[0].score < cfg.minGridScore) return [];
  const grids = best.map((g) => refine(img, g, cfg)).sort((p, q) => p.bbox[0] - q.bbox[0]);
  if (grids.length === 2) {
    grids[0].side = "left";
    grids[1].side = "right";
  }
  return grids;
}
