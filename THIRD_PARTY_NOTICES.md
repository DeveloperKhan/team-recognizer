# Third-party notices

The code in this repository is MIT-licensed (see `LICENSE`). It ships or downloads the
following third-party components, each under its own license.

## Models

| Component | Where | License |
| --- | --- | --- |
| DINOv2 ViT-S/14 weights (Meta AI), exported to ONNX | `models/dinov2_vits14.onnx`; the browser package's `dinov2.*.onnx` | Apache-2.0, https://github.com/facebookresearch/dinov2 |
| PaddleOCR PP-OCRv4 text detection and recognition models, as distributed by RapidOCR | the browser package's `ocr-det.onnx`, `ocr-rec.onnx`; installed with `rapidocr-onnxruntime` | Apache-2.0, https://github.com/PaddlePaddle/PaddleOCR, https://github.com/RapidAI/RapidOCR |

The species head (`models/species_head.npz`) and shadow probe (`models/shadow.npz`) were trained
for this project and are covered by `LICENSE`.

## Libraries (installed, not vendored)

onnxruntime and onnxruntime-web (MIT), timm (Apache-2.0), PyTorch (BSD-3-Clause), OpenCV
(Apache-2.0), NumPy (BSD-3-Clause), Pillow (MIT-CMU), FastAPI (MIT), rapidfuzz (MIT).

## Data

- Species ids come from PvPoke's gamemaster (https://github.com/pvpoke/pvpoke, MIT).
- Sprite ids and names come from Pokémon Showdown and Smogon's sprite index.
- Reference sprites are downloaded when building the gallery (`recognizer build-refs`): Pokémon
  Showdown's sprites, and Pokémon GO's icon art from PokeMiners' mined game assets
  (https://github.com/PokeMiners/pogo_assets, pinned to a commit). They are not included in this
  repository or the npm package; only the models trained on them are.

## Trademarks

Pokémon and all related names, characters and images are trademarks and © of Nintendo,
Creatures Inc. and GAME FREAK inc. / The Pokémon Company. This project is unofficial and is not
affiliated with, endorsed or sponsored by any of them.
