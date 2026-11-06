# @dracoviz/team-recognizer

Reads Pokémon GO tournament teams from screenshots of the 2027 Play! Pokémon broadcast overlay,
entirely in the browser. For each team in the image you get the six Pokémon (shadow forms
included), the player's name and their country. The models run in a Web Worker, on WebGPU where
available and otherwise on WebAssembly, so images never leave the device.

[Demo](https://team-recognizer.vercel.app) · [Source, accuracy and the Python/HTTP versions](https://github.com/DeveloperKhan/team-recognizer)

```sh
npm install @dracoviz/team-recognizer
```

```js
import { createLocalRecognizer } from "@dracoviz/team-recognizer";

const recognizer = createLocalRecognizer();
const { backend } = await recognizer.ready;       // loads the models; "webgpu" or "wasm"
const { teams } = await recognizer.recognize(file); // a File or Blob (PNG, JPEG, WebP...)

for (const team of teams) {
  console.log(team.player, team.country, team.pokemon.map((p) => (p.shadow ? `${p.speciesId} (shadow)` : p.speciesId)));
}
```

To skip the bundler, import it from the CDN:
`import { createLocalRecognizer } from "https://cdn.jsdelivr.net/npm/@dracoviz/team-recognizer@0.2.0/dist/client.js"`.

## API

### `createLocalRecognizer(options?)`

Starts the worker and begins downloading the models. Create one recognizer and reuse it. Options:

| Option | Default | |
| --- | --- | --- |
| `baseUrl` | this version's files on jsDelivr | where this package's `dist/` and `models/` are served from |
| `ortBaseUrl` | the matching onnxruntime-web release on jsDelivr | where onnxruntime-web's `dist/` files are served from |
| `onProgress(loaded, total)` | | model download progress, in bytes |

It returns:

- **`ready`**: resolves to `{ backend, version }` once the models are loaded. It rejects if the
  browser can't run them (no WebAssembly, or the files can't be fetched), so you can fall back to
  a server.
- **`recognize(image)`**: resolves to `{ teams, elapsedMs, timings }`. Calls are queued and run
  one at a time.
- **`dispose()`**: stops the worker.

### Result

```ts
interface RecognizedTeam {
  side: "left" | "right" | null; // null for a crop of one team's panel
  player: string | null;         // as read off the overlay
  country: string | null;        // e.g. "US"
  nameBox: RelativeBox;
  pokemon: RecognizedPokemon[];  // six, in grid order
}

interface RecognizedPokemon {
  speciesId: string;             // PvPoke id without "_shadow", e.g. "ninetales_alolan"
  shadow: boolean;
  confidence: number;            // 0–1
  alternatives: Array<{ speciesId: string; confidence: number }>;
  box: RelativeBox;              // the tile
  shadowViolet: number;          // violet fraction of the tile's border
  shadowProbe: number;           // the learned shadow check's probability
}

type RelativeBox = [x0: number, y0: number, x1: number, y1: number]; // fractions of the image size
```

To show a sprite, `models/config.json` lists every class with its Pokémon Showdown sprite name
(`species.classes[].showdownSlug`). The image is then
`https://play.pokemonshowdown.com/sprites/home/<slug>.png`.

## Downloads and hosting

The first use downloads about 60 MB of models on WebGPU (40 MB on WebAssembly), plus the ONNX
runtime. Every URL is versioned, so the browser caches them for later visits.

- **Self-hosting:** serve the package's `dist/` and `models/` folders from your own origin and
  pass `baseUrl`. Serve `node_modules/onnxruntime-web/dist/` and pass `ortBaseUrl` to drop jsDelivr
  entirely. The runtime must be the release in `DEFAULT_ORT_BASE_URL`.
- **Content Security Policy:** with the default (CDN) URLs, the page needs `worker-src blob:` and
  `cdn.jsdelivr.net` in `script-src` and `connect-src`. Browsers only start workers from the page's
  own origin, so the client starts a tiny same-origin bootstrap that imports the CDN worker.
- **Threads:** WebAssembly uses several threads when the page is cross-origin isolated (COOP +
  COEP headers); otherwise it uses one.

## Development

This folder is the TypeScript port of the Python pipeline in the repository root.

```sh
npm install
uv run recognizer export-web   # from the repo root: models, config.json and test fixtures into web/models
npm test                       # image operations against OpenCV fixtures
npm run parity -- <screenshots> preds.json && uv run recognizer eval --predictions preds.json
npm run build                  # dist/: worker.js (bundles onnxruntime-web), client.js, .d.ts
npm run demo                   # the demo page's script, public/demo.js
```

| Module | Port of |
| --- | --- |
| `src/grid.ts` | `recognizer/grid.py` |
| `src/image.ts` | the OpenCV calls it relies on (INTER_AREA resize, 8-bit Lab/HSV) |
| `src/pipeline.ts`, `src/assign.ts` | `recognizer/pipeline.py`, `assign.py`, `forms.py` |
| `src/shadow.ts` | `recognizer/shadow.py` |
| `src/ocr.ts` | `recognizer/ocr.py` + RapidOCR's detector/recognizer steps (horizontal text only) |

MIT licensed. The bundled DINOv2 and PaddleOCR models are Apache-2.0; see `THIRD_PARTY_NOTICES.md`.
Not affiliated with Niantic, Nintendo or The Pokémon Company.
