/**
 * Main-thread API for the in-browser recognizer. The work happens in a Web Worker served from
 * `baseUrl` alongside the ONNX runtime and the models (see scripts/sync.mjs).
 *
 *   const recognizer = createLocalRecognizer({ baseUrl: "/team-recognizer/<version>/" });
 *   await recognizer.ready;                 // downloads models the first time
 *   const teams = await recognizer.recognize(file);
 */

import type { Backend, RecognizedTeam, WorkerRequest, WorkerResponse } from "./types";

export type { Backend, RecognizedPokemon, RecognizedTeam, RelativeBox } from "./types";

export interface LocalRecognizer {
  /** Resolves once the models are loaded; rejects if this browser can't run them. */
  ready: Promise<{ backend: Backend; version: string }>;
  recognize(image: Blob): Promise<{ teams: RecognizedTeam[]; elapsedMs: number }>;
  dispose(): void;
}

export interface LocalRecognizerOptions {
  baseUrl: string;
  /** Download progress while models load, in bytes. */
  onProgress?: (loaded: number, total: number) => void;
}

export function createLocalRecognizer({ baseUrl, onProgress }: LocalRecognizerOptions): LocalRecognizer {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const worker = new Worker(`${base}worker.js`, { type: "module" });
  const pending = new Map<number, { resolve: (v: { teams: RecognizedTeam[]; elapsedMs: number }) => void; reject: (e: Error) => void }>();
  let nextId = 1;
  let settle: { resolve: (v: { backend: Backend; version: string }) => void; reject: (e: Error) => void };
  const ready = new Promise<{ backend: Backend; version: string }>((resolve, reject) => (settle = { resolve, reject }));

  worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
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
        pending.get(message.id)?.resolve({ teams: message.teams, elapsedMs: message.elapsedMs });
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
    const error = new Error(event.message || "Recognizer worker failed");
    settle.reject(error);
    pending.forEach((p) => p.reject(error));
    pending.clear();
  };

  const send = (message: WorkerRequest) => worker.postMessage(message);
  send({ type: "init", baseUrl: base });

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
      pending.forEach((p) => p.reject(new Error("Recognizer disposed")));
      pending.clear();
    },
  };
}
