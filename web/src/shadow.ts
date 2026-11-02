/**
 * Shadow detection (port of recognizer/shadow.py): a violet-pixel check on each tile's border
 * band, falling back to a logistic probe on DINOv2 features of the band when the smoke has lost
 * its color (compressed crops, fainted tiles).
 */

import type { RecognizerConfig } from "./config.ts";
import { hsv, type RGBImage } from "./image.ts";

/** Share of the border band (rim excluded) that is violet. */
export function violetFraction(tile: RGBImage, colorBand: number): number {
  const { width: w, height: h, data } = tile;
  const bh = Math.max(1, Math.trunc(colorBand * h));
  const bw = Math.max(1, Math.trunc(colorBand * w));
  const rim = Math.max(1, Math.trunc(0.03 * h));
  let total = 0;
  let violet = 0;
  for (let y = 0; y < h; y++) {
    if (y < rim || y >= h - rim) continue;
    const inRows = y < bh || y >= h - bh;
    for (let x = 0; x < w; x++) {
      if (x < rim || x >= w - rim) continue;
      if (!inRows && x >= bw && x < w - bw) continue;
      const i = (y * w + x) * 3;
      const [H, S, V] = hsv(data[i], data[i + 1], data[i + 2]);
      total++;
      if (H >= 108 && H <= 165 && S >= 35 && V >= 35) violet++;
    }
  }
  return total ? violet / total : 0;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** The tile with everything but its border band painted over in the band's median color. */
export function bandOnly(tile: RGBImage, band: number): RGBImage {
  const { width: w, height: h, data } = tile;
  const bh = Math.max(1, Math.trunc(band * h));
  const bw = Math.max(1, Math.trunc(band * w));
  // np.concatenate of the four strips: corners are counted twice, as in Python.
  const channels: number[][] = [[], [], []];
  const take = (x: number, y: number) => {
    const i = (y * w + x) * 3;
    for (let c = 0; c < 3; c++) channels[c].push(data[i + c]);
  };
  for (let y = 0; y < bh; y++) for (let x = 0; x < w; x++) take(x, y);
  for (let y = h - bh; y < h; y++) for (let x = 0; x < w; x++) take(x, y);
  for (let y = 0; y < h; y++) for (let x = 0; x < bw; x++) take(x, y);
  for (let y = 0; y < h; y++) for (let x = w - bw; x < w; x++) take(x, y);
  const fill = channels.map((c) => Math.trunc(median(c)));
  const out = new Uint8Array(data);
  for (let y = bh; y < h - bh; y++) {
    for (let x = bw; x < w - bw; x++) {
      const i = (y * w + x) * 3;
      out[i] = fill[0];
      out[i + 1] = fill[1];
      out[i + 2] = fill[2];
    }
  }
  return { width: w, height: h, data: out };
}

export function probeProbability(feats: Float32Array, cfg: RecognizerConfig["shadow"]["probe"]): number {
  let z = cfg.intercept;
  for (let i = 0; i < feats.length; i++) z += ((feats[i] - cfg.mean[i]) / cfg.scale[i]) * cfg.coef[i];
  return 1 / (1 + Math.exp(-z));
}
