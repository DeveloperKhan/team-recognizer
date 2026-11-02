# team-recognizer

Identifies Pokémon GO tournament teams from screenshots of the 2027 Play! Pokémon broadcast
overlay: for every team grid in the image (one for a cropped player panel, two for a full
broadcast frame), the player's name and six Dracoviz speciesIds, shadow included. No LLMs: a
classical lattice search finds the grids, a frozen DINOv2 image model plus a small trained
classifier identifies sprites, and RapidOCR reads names.

## Setup

```sh
brew install uv                  # Python is managed by uv (3.11)
uv sync
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
| Species | `embed.py`, `train_head.py`, `assign.py` | Frozen DINOv2 ViT-S/14 features; a cosine classifier trained on synthetic + real labeled tiles; Hungarian assignment so a team has no duplicate species |
| Shadow | `shadow.py` | Logistic regression on DINOv2 features of each tile's border band (where the smoke shows), trained on real labeled tiles |
| Names | `ocr.py` | RapidOCR on the name bar above the grid, fuzzy-matched to the event roster |

The overlay's shape lives in `grid.STYLES`. If a future overlay changes the tile aspect ratio or
gutters, add a style measured from one clean frame, rerun `build-refs` with a few of its
screenshots, and label a few screenshots so the shadow classifier sees its smoke.
