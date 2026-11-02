/**
 * Image operations, matching the OpenCV calls the Python recognizer makes closely enough
 * that features computed here line up with the ones its models were trained on.
 *
 * Images are interleaved RGB, 8 bits per channel.
 */

export interface RGBImage {
  width: number;
  height: number;
  data: Uint8Array; // length width * height * 3
}

/** Python's round(): halves go to the even neighbour. */
export function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
}

export function crop(img: RGBImage, x0: number, y0: number, x1: number, y1: number): RGBImage {
  const w = Math.max(0, x1 - x0);
  const h = Math.max(0, y1 - y0);
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const src = ((y0 + y) * img.width + x0) * 3;
    out.set(img.data.subarray(src, src + w * 3), y * w * 3);
  }
  return { width: w, height: h, data: out };
}

/** Per-axis taps for cv2.resize(INTER_AREA): true area averaging when shrinking. */
function areaTaps(src: number, dst: number): Array<Array<[number, number]>> {
  const scale = src / dst;
  const taps: Array<Array<[number, number]>> = [];
  for (let d = 0; d < dst; d++) {
    const start = d * scale;
    const end = start + scale;
    const list: Array<[number, number]> = [];
    for (let s = Math.floor(start); s < Math.ceil(end) && s < src; s++) {
      const w = Math.min(end, s + 1) - Math.max(start, s);
      if (w > 1e-9) list.push([s, w / scale]);
    }
    taps.push(list);
  }
  return taps;
}

/**
 * Per-axis taps for cv2.resize(INTER_AREA) when enlarging: OpenCV switches to linear
 * interpolation with its own coefficients ("area mode"), sharper than plain bilinear.
 */
function areaModeLinearTaps(src: number, dst: number): Array<Array<[number, number]>> {
  const scale = src / dst;
  const inv = dst / src;
  const taps: Array<Array<[number, number]>> = [];
  for (let d = 0; d < dst; d++) {
    let s = Math.floor(d * scale);
    let f = d + 1 - (s + 1) * inv;
    f = f <= 0 ? 0 : f - Math.floor(f);
    if (s < 0) {
      f = 0;
      s = 0;
    }
    if (s >= src - 1) {
      f = 0;
      s = src - 1;
    }
    taps.push(f === 0 ? [[s, 1]] : [[s, 1 - f], [s + 1, f]]);
  }
  return taps;
}

/** Bilinear taps with half-pixel centres (cv2 INTER_LINEAR), for OCR crops. */
function linearTaps(src: number, dst: number): Array<Array<[number, number]>> {
  const scale = src / dst;
  const taps: Array<Array<[number, number]>> = [];
  for (let d = 0; d < dst; d++) {
    const x = (d + 0.5) * scale - 0.5;
    let s = Math.floor(x);
    let f = x - s;
    if (s < 0) {
      s = 0;
      f = 0;
    }
    if (s >= src - 1) {
      s = src - 1;
      f = 0;
    }
    taps.push(f === 0 ? [[s, 1]] : [[s, 1 - f], [s + 1, f]]);
  }
  return taps;
}

function separable(
  img: RGBImage,
  width: number,
  height: number,
  xTaps: Array<Array<[number, number]>>,
  yTaps: Array<Array<[number, number]>>
): RGBImage {
  const { width: sw, height: sh, data } = img;
  const tmp = new Float32Array(width * sh * 3);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0;
      for (const [s, w] of xTaps[x]) {
        const i = (y * sw + s) * 3;
        r += data[i] * w;
        g += data[i + 1] * w;
        b += data[i + 2] * w;
      }
      const o = (y * width + x) * 3;
      tmp[o] = r;
      tmp[o + 1] = g;
      tmp[o + 2] = b;
    }
  }
  const out = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const taps = yTaps[y];
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0;
      for (const [s, w] of taps) {
        const i = (s * width + x) * 3;
        r += tmp[i] * w;
        g += tmp[i + 1] * w;
        b += tmp[i + 2] * w;
      }
      const o = (y * width + x) * 3;
      out[o] = Math.min(255, Math.max(0, Math.round(r)));
      out[o + 1] = Math.min(255, Math.max(0, Math.round(g)));
      out[o + 2] = Math.min(255, Math.max(0, Math.round(b)));
    }
  }
  return { width, height, data: out };
}

/**
 * Repeated INTER_AREA resizes of one image. Shrinking is area averaging over piecewise-constant
 * pixels, which is exactly bilinear interpolation of the image's integral, so every target size
 * costs O(output) after one O(input) pass. Enlarging falls back to resizeArea.
 */
export class AreaSampler {
  private img: RGBImage;
  private integral: Float64Array; // (H+1) x (W+1) x 3

  constructor(img: RGBImage) {
    this.img = img;
    const { width: W, height: H, data } = img;
    const W1 = W + 1;
    const I = new Float64Array((H + 1) * W1 * 3);
    for (let y = 0; y < H; y++) {
      let r = 0, g = 0, b = 0;
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 3;
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
        const o = ((y + 1) * W1 + x + 1) * 3, p = (y * W1 + x + 1) * 3;
        I[o] = I[p] + r;
        I[o + 1] = I[p + 1] + g;
        I[o + 2] = I[p + 2] + b;
      }
    }
    this.integral = I;
  }

  resize(width: number, height: number): RGBImage {
    const { width: W, height: H } = this.img;
    if (width > W || height > H) return resizeArea(this.img, width, height);
    const W1 = W + 1;
    const I = this.integral;
    const sx = W / width, sy = H / height;
    // Integral at fractional coordinates: piecewise-linear between pixel corners along each axis.
    const xs = new Float64Array(width + 1).map((_, i) => Math.min(i * sx, W));
    const ys = new Float64Array(height + 1).map((_, i) => Math.min(i * sy, H));
    const out = new Uint8Array(width * height * 3);
    const at = (x: number, y: number, c: number) => {
      const x0 = Math.min(Math.floor(x), W - 1), y0 = Math.min(Math.floor(y), H - 1);
      const fx = x - x0, fy = y - y0;
      const i00 = (y0 * W1 + x0) * 3 + c, i01 = i00 + 3, i10 = i00 + W1 * 3, i11 = i10 + 3;
      return (I[i00] * (1 - fx) + I[i01] * fx) * (1 - fy) + (I[i10] * (1 - fx) + I[i11] * fx) * fy;
    };
    const area = sx * sy;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const o = (y * width + x) * 3;
        for (let c = 0; c < 3; c++) {
          const v = (at(xs[x + 1], ys[y + 1], c) - at(xs[x], ys[y + 1], c) - at(xs[x + 1], ys[y], c) + at(xs[x], ys[y], c)) / area;
          out[o + c] = Math.min(255, Math.max(0, Math.round(v)));
        }
      }
    }
    return { width, height, data: out };
  }
}

/** cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA). */
export function resizeArea(img: RGBImage, width: number, height: number): RGBImage {
  if (width === img.width && height === img.height) return img;
  const shrinking = img.width >= width && img.height >= height;
  const taps = shrinking ? areaTaps : areaModeLinearTaps;
  return separable(img, width, height, taps(img.width, width), taps(img.height, height));
}

/** cv2.resize(img, (width, height)) with the default bilinear interpolation. */
export function resizeLinear(img: RGBImage, width: number, height: number): RGBImage {
  if (width === img.width && height === img.height) return img;
  return separable(img, width, height, linearTaps(img.width, width), linearTaps(img.height, height));
}

// sRGB → linear, as a lookup over the 256 byte values.
const LINEAR = new Float64Array(256).map((_, v) => {
  const x = v / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
});

function labF(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
}

/**
 * cv2.cvtColor(img, COLOR_RGB2LAB) for 8-bit images, as float32: L scaled to 0-255,
 * a and b offset by 128.
 */
export function rgbToLab(img: RGBImage): Float32Array {
  const n = img.width * img.height;
  const out = new Float32Array(n * 3);
  const d = img.data;
  for (let i = 0; i < n; i++) {
    const r = LINEAR[d[i * 3]], g = LINEAR[d[i * 3 + 1]], b = LINEAR[d[i * 3 + 2]];
    const X = (0.412453 * r + 0.35758 * g + 0.180423 * b) / 0.950456;
    const Y = 0.212671 * r + 0.71516 * g + 0.072169 * b;
    const Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
    const fx = labF(X), fy = labF(Y), fz = labF(Z);
    const L = Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y;
    out[i * 3] = Math.min(255, Math.max(0, Math.round((L * 255) / 100)));
    out[i * 3 + 1] = Math.min(255, Math.max(0, Math.round(500 * (fx - fy) + 128)));
    out[i * 3 + 2] = Math.min(255, Math.max(0, Math.round(200 * (fy - fz) + 128)));
  }
  return out;
}

/** cv2 RGB2HSV for 8-bit pixels: H in [0, 180), S and V in [0, 255]. */
export function hsv(r: number, g: number, b: number): [number, number, number] {
  const v = Math.max(r, g, b);
  const diff = v - Math.min(r, g, b);
  const s = v === 0 ? 0 : Math.round((diff * 255) / v);
  let h = 0;
  if (diff !== 0) {
    if (v === r) h = (60 * (g - b)) / diff;
    else if (v === g) h = 120 + (60 * (b - r)) / diff;
    else h = 240 + (60 * (r - g)) / diff;
    if (h < 0) h += 360;
  }
  return [Math.round(h / 2) % 180, s, v];
}

/** Pad by replicating the outermost pixels (cv2.copyMakeBorder BORDER_REPLICATE), any channel count. */
export function padReplicate(data: Float32Array, width: number, height: number, channels: number, pad: number) {
  const W = width + 2 * pad;
  const H = height + 2 * pad;
  const out = new Float32Array(W * H * channels);
  for (let y = 0; y < H; y++) {
    const sy = Math.min(height - 1, Math.max(0, y - pad));
    for (let x = 0; x < W; x++) {
      const sx = Math.min(width - 1, Math.max(0, x - pad));
      const s = (sy * width + sx) * channels;
      const o = (y * W + x) * channels;
      for (let c = 0; c < channels; c++) out[o + c] = data[s + c];
    }
  }
  return { data: out, width: W, height: H };
}
