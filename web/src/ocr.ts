/**
 * Player names and country tags (port of recognizer/ocr.py, which runs RapidOCR).
 *
 * Same PaddleOCR detector and recognizer models; the RapidOCR steps around them are re-done
 * here for horizontal text only: DB post-processing keeps axis-aligned boxes (connected
 * components of the probability map, expanded by the unclip distance) instead of rotated
 * polygons, and there is no orientation classifier.
 */

import type { RecognizerConfig } from "./config.ts";
import { crop, pyRound, resizeLinear, type RGBImage } from "./image.ts";
import type { Backend } from "./models.ts";

export interface TextResult {
  text: string;
  confidence: number;
  box: [number, number, number, number]; // in the OCR'd crop's pixels
}

// Name bars also hold the match score and a country code; neither is part of the name.
const NOISE = /^(\d+|[A-Z]{2})$/;
const RESULT_SUFFIX = /\s*\((W|L)\)\s*$/i;

/** RapidOCR's letterbox for very wide or short crops: pad top and bottom with black. */
function letterbox(img: RGBImage): RGBImage {
  const ratio = 8;
  const minHeight = 30;
  if (img.height > minHeight && img.width / img.height <= ratio) return img;
  const newH = Math.max(Math.trunc(img.width / ratio), minHeight) * 2;
  const pad = Math.trunc(Math.abs(newH - img.height) / 2);
  const out = new Uint8Array(img.width * (img.height + 2 * pad) * 3);
  out.set(img.data, pad * img.width * 3);
  return { width: img.width, height: img.height + 2 * pad, data: out };
}

function toCHW(img: RGBImage, width = img.width): Float32Array {
  // (x / 255 - 0.5) / 0.5; columns beyond the image (padding) stay 0.
  const { height: h, data } = img;
  const out = new Float32Array(3 * h * width);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < img.width; x++) {
      const i = (y * img.width + x) * 3;
      for (let c = 0; c < 3; c++) out[c * h * width + y * width + x] = data[i + c] / 127.5 - 1;
    }
  }
  return out;
}

export class Ocr {
  private backend: Backend;
  private cfg: RecognizerConfig["ocr"];
  private characters: string[];
  private codes: Set<string>;

  constructor(backend: Backend, keys: string[], cfg: RecognizerConfig["ocr"]) {
    this.backend = backend;
    this.cfg = cfg;
    this.characters = ["blank", ...keys, " "];
    this.codes = new Set(cfg.countryCodes);
  }

  /** Text boxes (DB post-processing, horizontal text) in `img`'s pixels. */
  private async detect(img: RGBImage): Promise<Array<[number, number, number, number]>> {
    const { minSide, thresh, boxThresh, unclipRatio } = this.cfg.det;
    const ratio = Math.min(img.width, img.height) < minSide ? minSide / Math.min(img.width, img.height) : 1;
    const round32 = (v: number) => Math.max(pyRound(v / 32) * 32, 32);
    const dw = round32(Math.trunc(img.width * ratio));
    const dh = round32(Math.trunc(img.height * ratio));
    const input = resizeLinear(img, dw, dh);
    const prob = await this.backend.run("det", toCHW(input), [1, 3, dh, dw]);

    // Threshold, then a 2x2 dilation as DB post-processing does.
    const mask = new Uint8Array(dw * dh);
    for (let i = 0; i < mask.length; i++) mask[i] = prob[i] > thresh ? 1 : 0;
    const dilated = new Uint8Array(mask);
    for (let y = 0; y < dh; y++)
      for (let x = 0; x < dw; x++)
        if (mask[y * dw + x]) {
          if (x + 1 < dw) dilated[y * dw + x + 1] = 1;
          if (y + 1 < dh) dilated[(y + 1) * dw + x] = 1;
          if (x + 1 < dw && y + 1 < dh) dilated[(y + 1) * dw + x + 1] = 1;
        }

    // Connected components (8-neighbour) → boxes.
    const label = new Int32Array(dw * dh);
    const boxes: Array<[number, number, number, number]> = [];
    const stack: number[] = [];
    let next = 0;
    for (let start = 0; start < dilated.length; start++) {
      if (!dilated[start] || label[start]) continue;
      next++;
      let x0 = dw, y0 = dh, x1 = 0, y1 = 0;
      stack.push(start);
      label[start] = next;
      while (stack.length) {
        const p = stack.pop()!;
        const px = p % dw, py = (p - px) / dw;
        x0 = Math.min(x0, px); x1 = Math.max(x1, px);
        y0 = Math.min(y0, py); y1 = Math.max(y1, py);
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const nx = px + dx, ny = py + dy;
            if (nx < 0 || ny < 0 || nx >= dw || ny >= dh) continue;
            const q = ny * dw + nx;
            if (dilated[q] && !label[q]) {
              label[q] = next;
              stack.push(q);
            }
          }
      }
      const w = x1 - x0 + 1, h = y1 - y0 + 1;
      if (Math.min(w, h) < 3) continue;
      // "fast" score: mean probability inside the box.
      let sum = 0;
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) sum += prob[y * dw + x];
      if (sum / (w * h) < boxThresh) continue;
      // Unclip: grow by area * ratio / perimeter on every side.
      const d = (w * h * unclipRatio) / (2 * (w + h));
      const ex0 = x0 - d, ey0 = y0 - d, ex1 = x1 + 1 + d, ey1 = y1 + 1 + d;
      if (Math.min(ex1 - ex0, ey1 - ey0) < 5) continue;
      const sx = img.width / dw, sy = img.height / dh;
      const clip = (v: number, hi: number) => Math.min(Math.max(Math.round(v), 0), hi);
      boxes.push([clip(ex0 * sx, img.width), clip(ey0 * sy, img.height), clip(ex1 * sx, img.width), clip(ey1 * sy, img.height)]);
    }
    // Top to bottom, then left to right within a line (RapidOCR's sorted_boxes).
    boxes.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    for (let i = 0; i < boxes.length - 1; i++)
      for (let j = i; j >= 0; j--) {
        if (Math.abs(boxes[j + 1][1] - boxes[j][1]) < 10 && boxes[j + 1][0] < boxes[j][0]) {
          [boxes[j], boxes[j + 1]] = [boxes[j + 1], boxes[j]];
        } else break;
      }
    return boxes.filter((b) => b[2] - b[0] > 3 && b[3] - b[1] > 3);
  }

  private async recognize(images: RGBImage[]): Promise<Array<{ text: string; confidence: number }>> {
    if (!images.length) return [];
    const { height, minWidth } = this.cfg.rec;
    const maxRatio = Math.max(minWidth / height, ...images.map((im) => im.width / im.height));
    const width = Math.trunc(height * maxRatio);
    const batch = new Float32Array(images.length * 3 * height * width);
    images.forEach((im, n) => {
      const w = Math.min(width, Math.ceil(height * (im.width / im.height)));
      batch.set(toCHW(resizeLinear(im, w, height), width), n * 3 * height * width);
    });
    const probs = await this.backend.run("rec", batch, [images.length, 3, height, width]);
    const C = this.characters.length;
    const T = probs.length / images.length / C;
    return images.map((_, n) => {
      let text = "";
      const confs: number[] = [];
      let prev = -1;
      for (let t = 0; t < T; t++) {
        const off = (n * T + t) * C;
        let best = 0;
        for (let c = 1; c < C; c++) if (probs[off + c] > probs[off + best]) best = c;
        if (best !== 0 && best !== prev) {
          text += this.characters[best];
          confs.push(probs[off + best]);
        }
        prev = best;
      }
      return { text, confidence: confs.length ? confs.reduce((s, v) => s + v, 0) / confs.length : 0 };
    });
  }

  /** All text in a region of the image (RapidOCR on the crop, upscaled to `height`). */
  async read(img: RGBImage, box: [number, number, number, number], height: number): Promise<TextResult[]> {
    let [x0, y0, x1, y1] = box.map(pyRound);
    x0 = Math.max(0, x0); y0 = Math.max(0, y0);
    x1 = Math.min(img.width, x1); y1 = Math.min(img.height, y1);
    if (y1 - y0 < 8 || x1 - x0 < 16) return [];
    let region = crop(img, x0, y0, x1, y1);
    const scale = height / region.height;
    if (scale > 1) region = resizeLinear(region, pyRound(region.width * scale), pyRound(region.height * scale));
    region = letterbox(region);
    const boxes = await this.detect(region);
    const texts = await this.recognize(boxes.map(([a, b, c, d]) => crop(region, a, b, c, d)));
    return texts.map((t, i) => ({ ...t, box: boxes[i] }));
  }

  /** The player's name in the name bar, or null (same filtering as ocr.read_name). */
  async readName(img: RGBImage, box: [number, number, number, number]): Promise<string | null> {
    let best: { width: number; text: string } | null = null;
    for (const r of await this.read(img, box, this.cfg.nameHeight)) {
      const text = r.text.trim().replace(RESULT_SUFFIX, "");
      if (r.confidence < this.cfg.minConfidence || !text || NOISE.test(text)) continue;
      const width = r.box[2] - r.box[0];
      if (!best || width > best.width || (width === best.width && text > best.text)) best = { width, text };
    }
    return best?.text ?? null;
  }

  /** The country tag above the name bar, or null (absent on tight crops). */
  async readCountry(img: RGBImage, box: [number, number, number, number]): Promise<string | null> {
    for (const r of await this.read(img, box, this.cfg.countryHeight)) {
      const code = r.text.toUpperCase().replace(/[^A-Z]/g, "");
      if (r.confidence >= this.cfg.countryMinConfidence && this.codes.has(code)) return code;
    }
    return null;
  }
}
