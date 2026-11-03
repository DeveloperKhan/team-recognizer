/**
 * Main-thread API for the in-browser recognizer. The work happens in a Web Worker that loads the
 * models from `baseUrl` (by default this package's own files on the jsDelivr CDN) and the ONNX
 * runtime from `ortBaseUrl`.
 *
 *   const recognizer = createLocalRecognizer();
 *   await recognizer.ready;                 // downloads the models the first time
 *   const { teams } = await recognizer.recognize(file);
 */

import type { Backend, RecognizedTeam, StageTiming, WorkerRequest, WorkerResponse } from "./types";

export type { Backend, RecognizedPokemon, RecognizedTeam, RelativeBox, StageTiming } from "./types";

// Replaced at build time (scripts/build.mjs).
declare const __PACKAGE_VERSION__: string;
declare const __ORT_VERSION__: string;

/** This package's version. */
export const VERSION: string = __PACKAGE_VERSION__;
/** Where the worker and models are served from unless `baseUrl` says otherwise. */
export const DEFAULT_BASE_URL: string = `https://cdn.jsdelivr.net/npm/@dracoviz/team-recognizer@${__PACKAGE_VERSION__}/`;
/** The onnxruntime-web release the worker bundles, whose WebAssembly files it loads. */
export const DEFAULT_ORT_BASE_URL: string = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${__ORT_VERSION__}/dist/`;

export interface RecognitionResult {
  teams: RecognizedTeam[];
  elapsedMs: number;
  /** Time spent in each stage (decode, grid search, image model, OCR...), in order. */
  timings: StageTiming[];
}

export interface LocalRecognizer {
  /** Resolves once the models are loaded; rejects if this browser can't run them. */
  ready: Promise<{ backend: Backend; version: string }>;
  recognize(image: Blob): Promise<RecognitionResult>;
  dispose(): void;
}

export interface LocalRecognizerOptions {
  /**
   * Folder holding this package's `dist/` and `models/`, to self-host them. Defaults to this
   * version's files on jsDelivr.
   */
  baseUrl?: string;
  /** Folder holding onnxruntime-web's `dist/` files. Defaults to the matching release on jsDelivr. */
  ortBaseUrl?: string;
  /** Download progress while the models load, in bytes. */
  onProgress?: (loaded: number, total: number) => void;
}

const absolute = (url: string) => {
  const href = new URL(url, globalThis.location?.href).href;
  return href.endsWith("/") ? href : `${href}/`;
};

/**
 * Browsers only start workers from the page's own origin. For a worker elsewhere (the CDN), start
 * a same-origin module that imports it instead; module imports may cross origins (with CORS).
 */
function startWorker(url: URL): { worker: Worker; release: () => void } {
  if (url.origin === globalThis.location?.origin) return { worker: new Worker(url, { type: "module" }), release() {} };
  const bootstrap = URL.createObjectURL(
    new Blob([`import ${JSON.stringify(url.href)};`], { type: "text/javascript" })
  );
  return { worker: new Worker(bootstrap, { type: "module" }), release: () => URL.revokeObjectURL(bootstrap) };
}

export function createLocalRecognizer(options: LocalRecognizerOptions = {}): LocalRecognizer {
  const { onProgress } = options;
  const base = absolute(options.baseUrl ?? DEFAULT_BASE_URL);
  const ortBase = absolute(options.ortBaseUrl ?? DEFAULT_ORT_BASE_URL);
  const { worker, release } = startWorker(new URL("dist/worker.js", base));
  const pending = new Map<number, { resolve: (v: RecognitionResult) => void; reject: (e: Error) => void }>();
  let nextId = 1;
  let settle: { resolve: (v: { backend: Backend; version: string }) => void; reject: (e: Error) => void };
  const ready = new Promise<{ backend: Backend; version: string }>((resolve, reject) => (settle = { resolve, reject }));
  ready.catch(() => {}); // callers that never await `ready` shouldn't see an unhandled rejection

  worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
    release();
    const message = event.data;
    switch (message.type) {
      case "progress":
        onProgress?.(message.loaded, message.total);
        break;
      case "log":
        console.debug(`[team-recognizer] ${message.message}`);
        break;
      case "ready":
        settle.resolve({ backend: message.backend, version: message.version });
        break;
      case "result":
        pending.get(message.id)?.resolve({ teams: message.teams, elapsedMs: message.elapsedMs, timings: message.timings });
        pending.delete(message.id);
        break;
      case "error":
        if (message.id === undefined) settle.reject(new Error(message.message));
        else {
          pending.get(message.id)?.reject(new Error(message.message));
          pending.delete(message.id);
        }
        break;
    }
  };
  worker.onerror = (event) => {
    release();
    const error = new Error(event.message || `Couldn't start the recognizer worker from ${base}`);
    settle.reject(error);
    pending.forEach((p) => p.reject(error));
    pending.clear();
  };

  const send = (message: WorkerRequest) => worker.postMessage(message);
  send({ type: "init", baseUrl: base, ortBaseUrl: ortBase });

  return {
    ready,
    recognize(image) {
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        send({ type: "recognize", id, image });
      });
    },
    dispose() {
      worker.terminate();
      release();
      const error = new Error("Recognizer disposed");
      settle.reject(error);
      pending.forEach((p) => p.reject(error));
      pending.clear();
    },
  };
}
