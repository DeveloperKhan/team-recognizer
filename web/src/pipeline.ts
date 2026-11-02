/**
 * Screenshot in, teams out (port of recognizer/pipeline.py and the API's response shape).
 */

import { assignTeam } from "./assign.ts";
import type { RecognizerConfig } from "./config.ts";
import { locate, type Grid } from "./grid.ts";
import { crop, pyRound, resizeArea, type RGBImage } from "./image.ts";
import type { Backend } from "./models.ts";
import { Ocr } from "./ocr.ts";
import { bandOnly, probeProbability, violetFraction } from "./shadow.ts";
import type { RecognizedPokemon, RecognizedTeam, RelativeBox } from "./types.ts";

export type { RecognizedPokemon, RecognizedTeam, RelativeBox };

/** Largest image side processed; bigger screenshots are scaled down first (as the API does). */
export const MAX_SIDE = 2400;

export class Recognizer {
  private config: RecognizerConfig;
  private head: Float32Array;
  private backend: Backend;
  private ocr: Ocr;

  constructor(config: RecognizerConfig, keys: string[], head: Float32Array, backend: Backend) {
    this.config = config;
    this.head = head;
    this.backend = backend;
    this.ocr = new Ocr(backend, keys, config.ocr);
  }

  private cropTile(img: RGBImage, box: [number, number, number, number]): RGBImage {
    const [x0, y0, x1, y1] = box;
    const dx = (x1 - x0) * this.config.tileInset;
    const dy = (y1 - y0) * this.config.tileInset;
    const xa = pyRound(Math.max(0, x0 + dx)), ya = pyRound(Math.max(0, y0 + dy));
    const xb = pyRound(Math.min(img.width, x1 - dx)), yb = pyRound(Math.min(img.height, y1 - dy));
    if (xb - xa < 4 || yb - ya < 4) return { width: 8, height: 8, data: new Uint8Array(8 * 8 * 3) };
    return crop(img, xa, ya, xb, yb);
  }

  /** L2-normalized DINOv2 features, one row per tile. */
  private async embed(tiles: RGBImage[]): Promise<Float32Array[]> {
    const { inputPx: S, mean, std, dim } = this.config.embed;
    const batch = new Float32Array(tiles.length * 3 * S * S);
    tiles.forEach((tile, n) => {
      const { data } = resizeArea(tile, S, S);
      const base = n * 3 * S * S;
      for (let i = 0; i < S * S; i++)
        for (let c = 0; c < 3; c++) batch[base + c * S * S + i] = (data[i * 3 + c] / 255 - mean[c]) / std[c];
    });
    const out = await this.backend.run("embed", batch, [tiles.length, 3, S, S]);
    return tiles.map((_, n) => {
      const f = out.slice(n * dim, (n + 1) * dim);
      const norm = Math.max(Math.hypot(...f), 1e-12);
      return f.map((v) => v / norm);
    });
  }

  /** Log-probabilities over the species classes for one tile's features. */
  private speciesLogProbs(f: Float32Array): Float32Array {
    const { classes, scale } = this.config.species;
    const dim = f.length;
    const logits = new Float32Array(classes.length);
    let max = -Infinity;
    for (let c = 0; c < classes.length; c++) {
      let dot = 0;
      for (let i = 0; i < dim; i++) dot += f[i] * this.head[c * dim + i];
      logits[c] = scale * dot;
      max = Math.max(max, logits[c]);
    }
    const probs = new Float64Array(classes.length);
    let sum = 0;
    for (let c = 0; c < logits.length; c++) {
      probs[c] = Math.exp(logits[c] - max);
      sum += probs[c];
    }
    for (let c = 0; c < probs.length; c++) probs[c] /= sum;
    // Forms that look the same on the overlay (Gourgeist sizes, Genesect drives) are one guess:
    // their probabilities move onto the representative form (see recognizer/forms.py).
    for (const [rep, others] of this.config.species.formGroups ?? []) {
      for (const o of others) {
        probs[rep] += probs[o];
        probs[o] = 0;
      }
    }
    return Float32Array.from(probs, (p) => Math.log(p + 1e-12));
  }

  async recognize(
    img: RGBImage,
    options: { readNames?: boolean; log?: (stage: string, ms: number) => void } = {}
  ): Promise<RecognizedTeam[]> {
    const { readNames = true, log } = options;
    let mark = performance.now();
    const lap = (stage: string) => {
      const now = performance.now();
      log?.(stage, now - mark);
      mark = now;
    };
    const { classes, topK } = this.config.species;
    const shadowCfg = this.config.shadow;
    const rel = ([x0, y0, x1, y1]: number[]): RelativeBox => {
      const clip = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 1e4) / 1e4;
      return [clip(x0 / img.width), clip(y0 / img.height), clip(x1 / img.width), clip(y1 / img.height)];
    };

    const teams: RecognizedTeam[] = [];
    const grids = locate(img, this.config.grid) as Grid[];
    lap(`grid search (${grids.length} grids)`);
    for (const grid of grids) {
      const boxes = grid.tiles();
      const tiles = boxes.map((b) => this.cropTile(img, b));
      const feats = await this.embed([...tiles, ...tiles.map((t) => bandOnly(t, shadowCfg.band))]);
      lap("image model (12 tiles)");
      const logp = feats.slice(0, 6).map((f) => this.speciesLogProbs(f));
      const picks = assignTeam(logp, topK);
      const pokemon = boxes.map((box, i): RecognizedPokemon => {
        const species = classes[picks[i]];
        const violet = violetFraction(tiles[i], shadowCfg.colorBand);
        const probe = probeProbability(feats[6 + i], shadowCfg.probe);
        const shadow =
          species.shadowEligible && (violet >= shadowCfg.colorThreshold || probe >= shadowCfg.probeThreshold);
        const order = Array.from(logp[i].keys()).sort((a, b) => logp[i][b] - logp[i][a]);
        return {
          speciesId: species.speciesId,
          shadow,
          confidence: Math.round(Math.exp(logp[i][picks[i]]) * 1000) / 1000,
          alternatives: order
            .filter((c) => c !== picks[i])
            .slice(0, 2)
            .map((c) => ({ speciesId: classes[c].speciesId, confidence: Math.round(Math.exp(logp[i][c]) * 1000) / 1000 })),
          box: rel(box),
          shadowViolet: Math.round(violet * 1000) / 1000,
          shadowProbe: Math.round(probe * 1000) / 1000,
        };
      });
      lap("species + shadow");
      teams.push({
        side: grid.side,
        player: readNames ? await this.ocr.readName(img, grid.nameBar()) : null,
        country: readNames ? await this.ocr.readCountry(img, grid.countryTag()) : null,
        nameBox: rel(grid.nameBar()),
        pokemon,
      });
      lap("OCR");
    }
    return teams;
  }
}

/** Scale an image down so its longer side is at most MAX_SIDE. */
export function limitSize(img: RGBImage): RGBImage {
  const longest = Math.max(img.width, img.height);
  if (longest <= MAX_SIDE) return img;
  const s = MAX_SIDE / longest;
  return resizeArea(img, Math.round(img.width * s), Math.round(img.height * s));
}
