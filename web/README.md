# team-recognizer/web

The recognizer as a browser module: the same pipeline as `recognizer/` (grid search, DINOv2
species head, shadow check, OCR), ported to TypeScript and run in a Web Worker with
onnxruntime-web. The image model runs on WebGPU in float16 where available, otherwise on
WebAssembly with int8 weights.

```sh
npm install
uv run recognizer export-web   # (from the repo root) models + config.json + test fixtures into web/models
npm test                       # image operations vs OpenCV fixtures
npm run parity -- ~/Desktop/screenshots preds.json && uv run recognizer eval --predictions preds.json
npm run build                  # dist/worker.js (bundles onnxruntime-web) + the ORT wasm file
npm run sync                   # copy into ../../play-graphic-tool (or a path given as the argument)
```

`sync` writes the worker, runtime and models to `public/team-recognizer/<hash>/` in the app
(the hash covers every file, so the folder can be cached forever) and `client.ts`, `types.ts`
and `version.ts` to `vendor/team-recognizer/`. The app then does:

```ts
import { createLocalRecognizer } from "@/vendor/team-recognizer/client";
import { RECOGNIZER_BASE_URL } from "@/vendor/team-recognizer/version";

const recognizer = createLocalRecognizer({ baseUrl: RECOGNIZER_BASE_URL, onProgress });
await recognizer.ready; // rejects if this browser can't run it
const { teams } = await recognizer.recognize(file);
```

`config.json` is generated from the Python modules (grid constants, species classes and form
groups, shadow thresholds and probe, OCR settings), so retraining or retuning in Python and
re-running `export-web` + `build` + `sync` is all it takes to update the browser.

| Module | Port of |
| --- | --- |
| `src/grid.ts` | `recognizer/grid.py` |
| `src/image.ts` | the OpenCV calls it relies on (INTER_AREA resize, 8-bit Lab/HSV) |
| `src/pipeline.ts`, `src/assign.ts` | `recognizer/pipeline.py`, `assign.py`, `forms.py` |
| `src/shadow.ts` | `recognizer/shadow.py` |
| `src/ocr.ts` | `recognizer/ocr.py` + RapidOCR's detector/recognizer steps (horizontal text only) |
