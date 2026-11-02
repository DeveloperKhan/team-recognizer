/**
 * Model execution, independent of where it runs: the browser worker passes onnxruntime-web,
 * the Node parity harness passes onnxruntime-node. Both expose the same InferenceSession and
 * Tensor API.
 */

import type * as OrtWeb from "onnxruntime-web";

export type ModelName = "embed" | "det" | "rec";

export interface Backend {
  run(model: ModelName, input: Float32Array, dims: number[]): Promise<Float32Array>;
}

type Ort = Pick<typeof OrtWeb, "Tensor">;
type Session = Pick<OrtWeb.InferenceSession, "run" | "inputNames" | "outputNames">;

export function createBackend(ort: Ort, sessions: Record<ModelName, Session>): Backend {
  return {
    async run(model, input, dims) {
      const session = sessions[model];
      const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input, dims) };
      const out = await session.run(feeds);
      return out[session.outputNames[0]].data as Float32Array;
    },
  };
}

/** The class weights (float16, row-major) as float32. */
export function decodeHead(buffer: ArrayBuffer): Float32Array {
  const half = new Uint16Array(buffer);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) {
    const h = half[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    out[i] =
      exp === 0
        ? sign * Math.pow(2, -14) * (frac / 1024)
        : exp === 31
          ? frac
            ? NaN
            : sign * Infinity
          : sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }
  return out;
}
