# team-recognizer

Reads Pokémon GO tournament teams from screenshots of the 2027 Play! Pokémon broadcast overlay.
For every team grid in the image (one for a cropped player panel, two for a full broadcast frame)
it returns the six Pokémon (shadow forms included), the player's name and their country. No LLMs:
a classical lattice search finds the grids, a frozen DINOv2 image model with a small trained
classifier identifies the sprites, and PaddleOCR reads the names.

**[Try it in your browser →](https://team-recognizer.vercel.app)** Everything runs on your
device, and your images aren't uploaded anywhere.

It runs three ways, all on the same models:

| | Where it runs | Good for |
| --- | --- | --- |
| [Browser package](#in-the-browser) `@dracoviz/team-recognizer` | the user's browser (WebGPU, or WebAssembly) | web tools: no server, images stay on the device |
| [Python CLI](#python) | your machine | batch jobs, training, evaluation |
| [HTTP API](#http-api) | a server you deploy (e.g. Vercel) | backends, or browsers that can't run the models |

## Output

```json
{
  "teams": [
    {
      "side": "left",
      "player": "PLAYERONE",
      "country": "US",
      "nameBox": [0.03, 0.31, 0.15, 0.36],
      "pokemon": [
        {
          "speciesId": "jumpluff",
          "shadow": true,
          "confidence": 0.97,
          "alternatives": [{ "speciesId": "electrode_hisuian", "confidence": 0.0 }],
          "box": [0.032, 0.42, 0.083, 0.53]
        }
      ]
    }
  ]
}
```

- `speciesId` uses [PvPoke](https://pvpoke.com)'s ids, without `_shadow`. `shadow` says whether
  it's the shadow form.
- `side` is `"left"` or `"right"` for a full broadcast frame and `null` for a single panel.
- `player` and `country` are read straight off the overlay, so the name comes out in the
  overlay's capitals. Map it to your own player list if you have one; the CLI can do that against
  an event roster.
- Boxes are `[x0, y0, x1, y1]` as fractions of the image's width and height.
- The browser build adds `shadowViolet` and `shadowProbe` to each Pokémon: the two scores behind
  the shadow decision, described under [How it works](#how-it-works).

## Accuracy

These numbers come from the held-out and training events; species and shadow are counted per
Pokémon. A "team" counts only if all six species and shadow flags are right.

| Event | Screenshots | Species | Shadow | Whole team right |
| --- | --- | --- | --- | --- |
| Baltimore 2027 | 14 teams | 100% | 100% | 14 / 14 |
| Gdańsk 2027 (held out of training) | 20 teams | 99.2% | 99.2% | 18 / 20 |
| Nice 2027 | 15 teams | 100% | 100% | 15 / 15 |
| Puebla 2027 | 32 teams | 99.5% | 100% | 31 / 32 |

Every grid and every name was found. Limits:
- Only the 2027 overlay is supported. Other layouts find no grid (see
  [a new overlay](#a-new-overlay)).
- The grid has to fill a fair part of the image. A full frame or any crop of one works; a frame
  shrunk into the corner of a much larger image isn't searched for.
- Indistinguishable cosmetic forms are reported as one form, e.g. Gourgeist sizes and Genesect drives.

## In the browser

```sh
npm install @dracoviz/team-recognizer
```

```js
import { createLocalRecognizer } from "@dracoviz/team-recognizer";

const recognizer = createLocalRecognizer({
  onProgress: (loaded, total) => console.log(`models: ${loaded} / ${total} bytes`),
});
const { backend } = await recognizer.ready; // "webgpu" or "wasm"; rejects if the browser can't run it
const { teams, elapsedMs } = await recognizer.recognize(file); // a File or Blob
```

You can skip the bundler: `import { createLocalRecognizer } from
"https://cdn.jsdelivr.net/npm/@dracoviz/team-recognizer@0.1.0/dist/client.js"`.

The work happens in a Web Worker, so the page stays responsive.
- **Downloads:** on first use it fetches about 60 MB of models on WebGPU (40 MB on WebAssembly),
  plus the ONNX runtime. Both come from jsDelivr by default, versioned and cached by the browser.
- **Speed:** a full frame takes about 1–2 s on WebGPU and 5–10 s on WebAssembly.
- **Self-hosting:** copy the package's `dist/` and `models/` folders somewhere and pass
  `baseUrl`. Copy onnxruntime-web's `dist/` and pass `ortBaseUrl` if you don't want jsDelivr at
  all.
- **Content Security Policy:** with the CDN, the page needs `worker-src blob:` (the worker is
  started from a small same-origin bootstrap that imports the CDN copy), plus
  `cdn.jsdelivr.net` in `script-src` and `connect-src`.

See [`web/README.md`](web/README.md) for the full API.

## Python

```sh
brew install uv     # or see https://docs.astral.sh/uv/; Python 3.12 is managed by uv
uv sync
uv run recognizer recognize screenshot.png          # or a folder of them
uv run recognizer recognize screenshot.png --json
```

Inference doesn't need PyTorch:
- The image model ships as `models/dinov2_vits14.onnx` (int8).
- The species head and the shadow probe ship as `.npz` files.

## HTTP API

`server.py` is a FastAPI app with two routes:

| Route | Request | Response |
| --- | --- | --- |
| `POST /recognize` | the image as multipart field `image`, or the raw bytes as the body | the [output](#output) above, plus `elapsedMs` |
| `GET /health` | — | a status check |

Set `RECOGNIZER_TOKEN` and clients must send `Authorization: Bearer <token>`; without it the
endpoint is open.

```sh
uv run uvicorn server:app --port 8000                             # local
curl -F image=@screenshot.png http://localhost:8000/recognize
vercel deploy                                                     # or deploy your fork on Vercel
```

The runtime dependencies come to about 350 MB, within Vercel's 500 MB Python limit. A Vercel
deployment also serves the demo page from `public/`.

## How it works

| Stage | Module | Method |
| --- | --- | --- |
| Find grids | `grid.py` | Multi-scale search for a 2×3 lattice of tile borders, scored by coherent contrast across each border. Both teams of a full frame are scored as a pair, then lines are snapped and the lattice refit |
| Reference sprites | `refs.py` | Pokémon Showdown's animated sprites (all frames) and HOME renders, one class per sprite (Megas excluded) |
| Synthetic tiles | `synth.py` | Sprites composited onto real tile backgrounds (sprite removed) or flat overlay fills, in bright, dimmed and fainted states, with stream-compression degradations |
| Species | `embed.py`, `train_head.py`, `assign.py` | Frozen DINOv2 ViT-S/14 features and a cosine classifier trained on synthetic and real labeled tiles. A Hungarian assignment keeps a team from having the same species twice |
| Shadow | `shadow.py` | A violet-pixel check on each tile's border band. When the smoke has lost its color (compressed crops, fainted tiles), a logistic regression on DINOv2 features of the band decides |
| Names | `ocr.py` | RapidOCR (PaddleOCR PP-OCRv4) on the name bar and the country tag, optionally fuzzy-matched to an event roster |

`web/` is the same pipeline ported to TypeScript. `recognizer export-web` generates its settings
from the Python modules, and its parity harness checks it against the Python results.

## Training and extending

Training needs the extra dependencies, `uv sync --extra train`. The steps:

```sh
uv run recognizer update-species                # species list from PvPoke, Showdown and Smogon
uv run recognizer build-refs --screens <folder> # download sprites, learn the overlay's tile colors, build the gallery
uv run recognizer label-screens                 # propose who is on screen in each screenshot (review it!)
uv run recognizer train --holdout <event>       # fit the shadow probe and species head
uv run recognizer eval --crops 5                # score, plus random re-crops of each team
uv run recognizer export-onnx                   # the int8 image model for inference
uv run recognizer export-web                    # the browser build's models and config
```

`build-refs` learns tile colors and backgrounds from the screenshots it's given. Don't give it
screenshots of an event you want to keep as an untouched test set.

### Your data

Labels and rosters live outside this repo, in a data folder: `../team-recognizer-data` if it
exists, or wherever `RECOGNIZER_DATA_DIR` points. Screenshots are read from `--screens`
(default `~/Desktop/screenshots`), one subfolder per event.

```
labels/events.json        {"Nice Regional 2027": "2027-nice", ...}   screenshot folder -> event
labels/screens.json       which player each grid in each screenshot belongs to
rosters/<event>.json      [{"name": "PlayerOne", "team": [{"speciesId": "altaria", "shadow": false}, ...]}, ...]
refs/extra/<speciesId>/   extra reference images for a species (PNG/WebP with transparency)
```

`screens.json` is keyed by the screenshot's path inside the screenshots folder:

```json
{
  "Nice Regional 2027/frame.png": { "event": "2027-nice", "left": "PlayerOne", "right": "PlayerTwo" },
  "Nice Regional 2027/panel.png": { "event": "2027-nice", "team": "PlayerThree" },
  "Other Event/panel.png": { "event": "other-event", "team": { "player": "PlayerFour", "pokemon": ["dunsparce", "ninetales_alolan_shadow", "..."] } }
}
```

`label-screens` proposes these entries by OCR-matching names against the rosters. Teams of
events without a roster can be written inline, as in the last line.

### Overlay art the references miss

Sometimes the overlay draws a Pokémon differently from Showdown's sprites. The white-flowered
Florges is one case. Drop a few images of that art into `refs/extra/<speciesId>/` in the data
folder, rebuild that class's gallery entries, and retrain.

### A new overlay

The overlay's geometry lives in `grid.STYLES` (tile aspect ratio and gutters). For a new layout:
1. Add a style measured from one clean frame.
2. Rerun `build-refs` with a few of its screenshots.
3. Label some screenshots so the shadow classifier sees its smoke, then retrain.

## Contributing

Reports of wrong results (with the screenshot) and pull requests are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md). Accuracy is measured on labeled screenshots that aren't
public, so changes to the recognition itself are evaluated by the maintainer before merging.
Security issues: see [SECURITY.md](SECURITY.md).

## License

The code and the trained species head and shadow probe are MIT licensed. See
[`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the bundled DINOv2
weights (Apache-2.0) and PaddleOCR models (Apache-2.0).

Sprites are downloaded from Pokémon Showdown at build time and aren't redistributed here. This
project isn't affiliated with Niantic, Scopely, Nintendo, Creatures, GAME FREAK or The Pokémon
Company. Pokémon and Pokémon GO are trademarks of their respective owners.
