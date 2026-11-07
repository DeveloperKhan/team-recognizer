// Compare the TypeScript grid search with Python's on real screenshots.
//   node --experimental-strip-types test/grid-parity.ts <python grids json>
import { readFileSync } from "node:fs";
import sharp from "sharp";
import { locate } from "../src/grid.ts";
import type { RecognizerConfig } from "../src/config.ts";

const config: RecognizerConfig = JSON.parse(readFileSync(new URL("../models/config.json", import.meta.url), "utf8"));
const expected: Record<string, Array<{ xs: number[]; ys: number[]; side: string }>> = JSON.parse(
  readFileSync(process.argv[2], "utf8")
);

export async function loadRgb(path: string) {
  // Pixels as stored, ignoring any embedded color profile, as OpenCV and the worker read them
  const { data, info } = await sharp(path, { ignoreIcc: true })
    .flatten({ background: "#000000" })
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data.buffer, data.byteOffset, data.length) };
}

let worst = 0, mismatched = 0, total = 0;
for (const [path, grids] of Object.entries(expected)) {
  const img = await loadRgb(path);
  const t = performance.now();
  const ours = locate(img, config.grid);
  const ms = performance.now() - t;
  total++;
  if (ours.length !== grids.length) {
    mismatched++;
    console.log(`COUNT ${path.split("/").slice(-2).join("/")}: ts=${ours.length} py=${grids.length}`);
    continue;
  }
  let diff = 0;
  ours.forEach((g, i) => {
    const tw = g.tileW;
    [...g.xs.map((v, k) => v - grids[i].xs[k]), ...g.ys.map((v, k) => v - grids[i].ys[k])].forEach(
      (d) => (diff = Math.max(diff, Math.abs(d) / tw))
    );
  });
  worst = Math.max(worst, diff);
  if (diff > 0.05) {
    mismatched++;
    console.log(`DIFF ${path.split("/").slice(-2).join("/")}: ${(diff * 100).toFixed(1)}% of a tile`);
  }
  if (total <= 3) console.log(`${path.split("/").pop()}: ${ms.toFixed(0)} ms`);
}
console.log(`${total} images, ${mismatched} differ, worst line offset ${(worst * 100).toFixed(2)}% of a tile`);
