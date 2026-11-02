# team-recognizer

Identifies Pokémon GO tournament teams from screenshots of the 2027 Play! Pokémon broadcast
overlay: for every team grid in the image (one for a cropped player panel, two for a full
broadcast frame), the player's name and six Dracoviz speciesIds, shadow included. No LLMs: a
classical lattice search finds the grids, a frozen DINOv2 image model plus a small trained
classifier identifies sprites, and RapidOCR reads names.

## Setup

```sh
brew install uv                  # Python is managed by uv (3.12)
uv sync --extra train            # inference-only installs can drop --extra train
node scripts/export_species.mjs  # species list + Showdown sprite ids (reads ../dracoviz-site)
uv run recognizer build-refs --screens ~/Desktop/screenshots/Nice\ Regional\ 2027
```

`build-refs` downloads Showdown's animated sprites and Pokémon HOME renders for every species,
learns the overlay's tile colors and backgrounds from the given screenshots, and renders + embeds
the synthetic gallery (`cache/gallery.npz`). Don't pass screenshots of events you want to keep
as an untouched test set.

## Use

```sh
uv run recognizer recognize path/to/screenshot.png            # or a folder
uv run recognizer recognize shot.png --event 2027-nice --json  # match names to the event roster
```

## HTTP API (Vercel)

`server.py` is a FastAPI app: `POST /recognize` with the image (multipart field `image`, or the
raw bytes) returns each team's player name, country code and six `{speciesId, shadow,
confidence, alternatives}`. Set `RECOGNIZER_TOKEN` in the Vercel project and send it as
`Authorization: Bearer <token>`; without the variable the endpoint is open (local development).

Inference needs no PyTorch: the image model ships as `models/dinov2_vits14.onnx` (int8, from
`recognizer export-onnx`), the species head and shadow probe as `.npz`. The runtime
dependencies are only the base `[project.dependencies]`, about 350 MB installed, within Vercel's
500 MB Python limit.

```sh
uv run uvicorn server:app --port 8000   # local
vercel deploy                           # Vercel picks up server.py and vercel.json
```

## Browser build

`web/` is the same pipeline ported to TypeScript, running in a Web Worker with onnxruntime-web
(WebGPU or WebAssembly). play-graphic-tool uses it so screenshots never leave the browser; the
HTTP API above is its fallback. `uv run recognizer export-web` writes the models and settings
it needs; see `web/README.md` for building and syncing it into the app.

Extra reference sprites for overlay art the downloaded references miss (e.g. the white-flowered
Florges) go in `refs/extra/<speciesId>/`; rebuild that class's gallery entries and retrain.

## Train and evaluate

```sh
# 1. Official rosters (needs GATSBY_SECRET_KEY; the key is never written anywhere)
node --env-file=../dracoviz-site/.env scripts/export_rosters.mjs 2027-nice 2027-baltimore 2027-puebla

# 2. Which player each grid belongs to: OCR + roster matching, written to labels/screens.json
#    (folders map to events in labels/events.json; review the output)
uv run recognizer label-screens

# 3. Fit the shadow classifier and species head on labeled tiles, holding out a test event
uv run recognizer train --holdout 2027-gdansk

# 4. Score everything, plus random re-crops of each full-frame team
uv run recognizer eval --crops 5
```

Ground truth is `data/rosters/<event>.json` (exported) and `labels/rosters/<event>.json`
(hand-labeled, for events not on Dracoviz yet: currently Gdańsk), plus `labels/screens.json`.

## How it works

| Stage | Module | Method |
| --- | --- | --- |
| Find grids | `grid.py` | Multi-scale search for a 2x3 lattice of tile borders, scored by coherent contrast across each border; both teams of a full frame scored as a pair; lines snapped and the lattice refit |
| Reference sprites | `refs.py` | Showdown `sprites/ani` GIF frames + HOME renders, one class per Dracoviz `sid` (Megas excluded) |
| Synthetic tiles | `synth.py` | Sprites composited onto real tile backgrounds (sprite removed) or flat overlay fills, in bright / see-through dimmed / gray states, with stream degradations |
| Species | `embed.py`, `train_head.py`, `assign.py` | Frozen DINOv2 ViT-S/14 features (ONNX at inference); a cosine classifier trained on synthetic + real labeled tiles; Hungarian assignment so a team has no duplicate species |
| Shadow | `shadow.py` | Violet-pixel check on each tile's border band; when the smoke has lost its color (compressed crops, fainted tiles) a logistic regression on DINOv2 features of the band, trained on real labeled tiles, decides |
| Names | `ocr.py` | RapidOCR on the name bar above the grid (fuzzy-matched to the event roster when given) and on the country tag above it |

The overlay's shape lives in `grid.STYLES`. If a future overlay changes the tile aspect ratio or
gutters, add a style measured from one clean frame, rerun `build-refs` with a few of its
screenshots, and label a few screenshots so the shadow classifier sees its smoke.
