/**
 * Web Worker that runs the recognizer off the page's main thread.
 *
 * Models load once from `baseUrl` (a versioned folder, so the browser can cache them forever),
 * the ONNX runtime's WebAssembly files from `ortBaseUrl`: the image model runs on WebGPU in
 * float16 when the browser has it, otherwise on WebAssembly with int8 weights; the OCR models
 * always run on WebAssembly.
 */

import * as ort from "onnxruntime-web/webgpu";
import type { RecognizerConfig } from "./config.ts";
import type { RGBImage } from "./image.ts";
import { createBackend, decodeHead } from "./models.ts";
import { limitSize, Recognizer } from "./pipeline.ts";
import type { Backend, StageTiming, WorkerRequest, WorkerResponse } from "./types.ts";

declare const self: DedicatedWorkerGlobalScope;

const post = (message: WorkerResponse) => self.postMessage(message);

let ready: Promise<Recognizer> | null = null;
// One recognition at a time: onnxruntime sessions can't run concurrently.
let queue: Promise<unknown> = Promise.resolve();

/** Download several files, reporting combined progress. */
async function fetchAll(urls: string[]): Promise<ArrayBuffer[]> {
  const totals = new Array<number>(urls.length).fill(0);
  const loaded = new Array<number>(urls.length).fill(0);
  const report = () => post({ type: "progress", loaded: sum(loaded), total: sum(totals) });
  return Promise.all(
    urls.map(async (url, i) => {
      const response = await fetch(url);
      if (!response.ok || !response.body) throw new Error(`${url}: HTTP ${response.status}`);
      totals[i] = Number(response.headers.get("content-length")) || 0;
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded[i] += value.length;
        report();
      }
      const out = new Uint8Array(loaded[i]);
      let offset = 0;
      for (const c of chunks) {
        out.set(c, offset);
        offset += c.length;
      }
      return out.buffer;
    })
  );
}

const sum = (values: number[]) => values.reduce((s, v) => s + v, 0);

async function hasWebGpu(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    return Boolean(gpu && (await gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function load(base: string, ortBase: string): Promise<Recognizer> {
  // The bundled onnxruntime-web JavaScript and these files must be the same release.
  ort.env.wasm.wasmPaths = ortBase;
  ort.env.logLevel = "error"; // not its warnings about ops placed on the CPU, which are expected
  // Threads need cross-origin isolation (SharedArrayBuffer); otherwise run single-threaded.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  let backend: Backend = (await hasWebGpu()) ? "webgpu" : "wasm";
  const embedFile = backend === "webgpu" ? "dinov2.fp16.onnx" : "dinov2.int8.onnx";
  const files = ["config.json", "ocr-keys.json", "head.bin", "ocr-det.onnx", "ocr-rec.onnx", embedFile];
  const [configBuf, keysBuf, headBuf, det, rec, embedModel] = await fetchAll(files.map((f) => `${base}models/${f}`));
  const decoder = new TextDecoder();
  const config: RecognizerConfig = JSON.parse(decoder.decode(configBuf));
  const keys: string[] = JSON.parse(decoder.decode(keysBuf));

  const wasm = { executionProviders: ["wasm"] };
  let embed: ort.InferenceSession;
  try {
    embed = await ort.InferenceSession.create(new Uint8Array(embedModel), {
      executionProviders: backend === "webgpu" ? ["webgpu"] : ["wasm"],
    });
  } catch (error) {
    if (backend !== "webgpu") throw error;
    // WebGPU is advertised but the session can't be created (driver/limits): use WebAssembly.
    backend = "wasm";
    const [int8] = await fetchAll([`${base}models/dinov2.int8.onnx`]);
    embed = await ort.InferenceSession.create(new Uint8Array(int8), wasm);
  }
  const sessions = {
    embed,
    det: await ort.InferenceSession.create(new Uint8Array(det), wasm),
    rec: await ort.InferenceSession.create(new Uint8Array(rec), wasm),
  };
  post({ type: "log", message: `ready: ${backend}, ${config.version}` });
  post({ type: "ready", backend, version: config.version });
  return new Recognizer(config, keys, decodeHead(headBuf), createBackend(ort, sessions));
}

/** Decode an image file to RGB, flattening any transparency onto black (as Python does). */
async function decode(image: Blob): Promise<RGBImage> {
  const bitmap = await createImageBitmap(image, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas unavailable");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const rgb = new Uint8Array(canvas.width * canvas.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    const a = rgba[i + 3] / 255;
    rgb[j] = Math.round(rgba[i] * a);
    rgb[j + 1] = Math.round(rgba[i + 1] * a);
    rgb[j + 2] = Math.round(rgba[i + 2] * a);
  }
  return { width: canvas.width, height: canvas.height, data: rgb };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  if (message.type === "init") {
    ready ??= load(message.baseUrl, message.ortBaseUrl);
    ready.catch((error) => post({ type: "error", message: String(error?.message ?? error) }));
    return;
  }
  if (message.type === "recognize") {
    const run = async () => {
      try {
        if (!ready) throw new Error("Recognizer not initialized");
        const recognizer = await ready;
        const start = performance.now();
        const image = limitSize(await decode(message.image));
        const timings: StageTiming[] = [];
        const log = (stage: string, ms: number) => {
          timings.push({ stage, ms: Math.round(ms) });
          post({ type: "log", message: `${stage}: ${Math.round(ms)} ms` });
        };
        log(`decode ${image.width}x${image.height}`, performance.now() - start);
        const teams = await recognizer.recognize(image, { log });
        post({ type: "result", id: message.id, teams, elapsedMs: Math.round(performance.now() - start), timings });
      } catch (error) {
        post({ type: "error", id: message.id, message: String((error as Error)?.message ?? error) });
      }
    };
    queue = queue.then(run, run);
  }
};
