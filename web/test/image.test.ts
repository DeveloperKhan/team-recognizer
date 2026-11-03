// Image operations against OpenCV reference outputs, written by `uv run recognizer write-fixtures`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { AreaSampler, resizeArea, rgbToLab, type RGBImage } from "../src/image.ts";

const dir = new URL("./fixtures/", import.meta.url);
if (!existsSync(new URL("meta.json", dir))) {
  throw new Error("No test fixtures: run `uv run recognizer write-fixtures` from the repository root first.");
}
const meta = JSON.parse(readFileSync(new URL("meta.json", dir), "utf8"));
const src: RGBImage = {
  width: meta.width,
  height: meta.height,
  data: new Uint8Array(readFileSync(new URL("src.rgb", dir))),
};

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>) {
  assert.equal(a.length, b.length);
  let worst = 0;
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    worst = Math.max(worst, d);
    total += d;
  }
  return { worst, mean: total / a.length };
}

test("rgbToLab matches cv2.COLOR_RGB2LAB", () => {
  const expected = new Uint8Array(readFileSync(new URL("lab.u8", dir)));
  const { worst, mean } = maxDiff(rgbToLab(src), expected);
  // OpenCV's 8-bit Lab uses fixed-point lookup tables; very dark saturated pixels land 1-2 off.
  assert.ok(worst <= 2, `worst ${worst}`);
  assert.ok(mean < 0.15, `mean ${mean}`);
});

for (const [w, h] of meta.area as Array<[number, number]>) {
  test(`resizeArea ${w}x${h} matches cv2 INTER_AREA`, () => {
    const expected = new Uint8Array(readFileSync(new URL(`area_${w}x${h}.rgb`, dir)));
    const { worst, mean } = maxDiff(resizeArea(src, w, h).data, expected);
    assert.ok(worst <= 2, `worst ${worst}`);
    assert.ok(mean < 0.2, `mean ${mean}`);
  });
}

for (const [w, h] of (meta.area as Array<[number, number]>).filter(([w, h]) => w <= meta.width && h <= meta.height)) {
  test(`AreaSampler ${w}x${h} matches cv2 INTER_AREA`, () => {
    const expected = new Uint8Array(readFileSync(new URL(`area_${w}x${h}.rgb`, dir)));
    const { worst, mean } = maxDiff(new AreaSampler(src).resize(w, h).data, expected);
    assert.ok(worst <= 2, `worst ${worst}`);
    assert.ok(mean < 0.2, `mean ${mean}`);
  });
}
