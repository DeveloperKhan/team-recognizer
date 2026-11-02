"""Export everything the browser build (web/) needs, from the same artifacts Python uses.

    uv run recognizer export-web

writes web/models/:
  dinov2.int8.onnx   image model for the WebAssembly backend (models/dinov2_vits14.onnx)
  dinov2.fp16.onnx   image model for WebGPU (float16 weights, float32 inputs/outputs)
  head.bin           species head: float16 class weights, row-major (classes x 384)
  ocr-det.onnx       PaddleOCR text detector, as RapidOCR ships it
  ocr-rec.onnx       PaddleOCR text recognizer
  ocr-keys.json      the recognizer's character list (read from the model's metadata)
  config.json        grid constants, species classes, shadow and OCR settings

Constants are read from the Python modules rather than restated, so the two implementations
can't drift apart silently.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import embed, forms, grid, ocr, pipeline, shadow, train_head
from .paths import CACHE, ROOT
from .refs import sprite_classes

WEB_MODELS = ROOT / "web" / "models"


def _fp16_model(dest: Path) -> None:
    """DINOv2 exported in half precision, with casts at the input and output so callers still
    pass and receive float32. (Converting the float32 ONNX graph instead leaves its Cast nodes
    inconsistent, or never finishes.)"""
    import numpy as np
    import onnx
    import timm
    import torch
    from onnx import TensorProto, helper

    model = timm.create_model(embed.MODEL_NAME, pretrained=True, num_classes=0, img_size=embed.INPUT_PX)
    model = model.eval().half()
    dummy = torch.from_numpy(embed.prepare([np.zeros((embed.INPUT_PX, embed.INPUT_PX, 3), np.uint8)] * 2)).half()
    tmp = CACHE / "dinov2_vits14.half.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(tmp),
        input_names=["pixels16"],
        output_names=["features16"],
        dynamic_axes={"pixels16": {0: "batch"}, "features16": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    m = onnx.load(str(tmp))
    g = m.graph
    g.node.insert(0, helper.make_node("Cast", ["pixels"], ["pixels16"], to=TensorProto.FLOAT16, name="input_to_fp16"))
    g.node.append(helper.make_node("Cast", ["features16"], ["features"], to=TensorProto.FLOAT, name="output_to_fp32"))
    g.input.remove(g.input[0])
    g.output.remove(g.output[0])
    px = embed.INPUT_PX
    g.input.append(helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["batch", 3, px, px]))
    g.output.append(helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", 384]))
    onnx.save(m, str(dest))


def _git_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "dev"


def export() -> dict:
    WEB_MODELS.mkdir(parents=True, exist_ok=True)

    # Image model: int8 for WebAssembly (as served by the API), float16 for WebGPU.
    shutil.copyfile(embed.ONNX_FILE, WEB_MODELS / "dinov2.int8.onnx")
    _fp16_model(WEB_MODELS / "dinov2.fp16.onnx")

    # Species head and the class list it scores.
    weights, classes = train_head.load()
    weights.astype("<f2").tofile(WEB_MODELS / "head.bin")
    by_sid = {sid: min(members, key=lambda s: len(s.speciesId)) for sid, members in sprite_classes().items()}
    species = [
        {"sid": int(sid), "speciesId": by_sid[int(sid)].speciesId, "shadowEligible": by_sid[int(sid)].shadowEligible}
        for sid in classes
    ]

    # OCR models, exactly as RapidOCR runs them.
    import onnx
    import rapidocr_onnxruntime

    ocr_dir = Path(rapidocr_onnxruntime.__file__).parent / "models"
    shutil.copyfile(ocr_dir / "ch_PP-OCRv4_det_infer.onnx", WEB_MODELS / "ocr-det.onnx")
    shutil.copyfile(ocr_dir / "ch_PP-OCRv4_rec_infer.onnx", WEB_MODELS / "ocr-rec.onnx")
    rec_meta = {p.key: p.value for p in onnx.load(str(ocr_dir / "ch_PP-OCRv4_rec_infer.onnx")).metadata_props}
    characters = rec_meta["character"].splitlines()
    (WEB_MODELS / "ocr-keys.json").write_text(json.dumps(characters, ensure_ascii=False))

    probe = dict(np.load(shadow.MODEL_FILE))
    config = {
        "version": _git_version(),
        "grid": {
            "styles": [
                {"name": s.name, "aspect": s.aspect, "gapX": s.gap_x, "gapY": s.gap_y} for s in grid.STYLES
            ],
            "tilePx": grid.TILE_PX,
            "refinePx": grid.REFINE_PX,
            "scaleStep": grid.SCALE_STEP,
            "minTilePx": grid.MIN_TILE_PX,
            "minTileFrac": grid.MIN_TILE_FRAC,
            "scaleAnchorFrac": grid.SCALE_ANCHOR_FRAC,
            "edgeSpan": list(grid.EDGE_SPAN),
            "strip": grid.STRIP,
            "contrastCap": grid.CONTRAST_CAP,
            "textureScale": grid.TEXTURE_SCALE,
            "minScore": grid.MIN_SCORE,
            "minGridScore": grid.MIN_GRID_SCORE,
            "sizePrior": grid.SIZE_PRIOR,
            "secondGridRatio": grid.SECOND_GRID_RATIO,
            "pairCandidates": grid.PAIR_CANDIDATES,
        },
        "tileInset": pipeline.TILE_INSET,
        "embed": {
            "inputPx": embed.INPUT_PX,
            "mean": embed.MEAN.ravel().tolist(),
            "std": embed.STD.ravel().tolist(),
            "dim": int(weights.shape[1]),
        },
        "species": {
            "scale": train_head.SCALE,
            "topK": 10,
            "classes": species,
            # Indistinguishable forms: sum the member columns' probabilities into the first.
            "formGroups": [
                [rep, [c for c in cols if c != rep]]
                for rep, cols in forms.merge_plan([c["speciesId"] for c in species])
            ],
        },
        "shadow": {
            "band": shadow.BAND,
            "colorBand": shadow.COLOR_BAND,
            "colorThreshold": shadow.COLOR_THRESHOLD,
            "probeThreshold": shadow.PROBE_THRESHOLD,
            "probe": {
                "mean": probe["mean"].tolist(),
                "scale": probe["scale"].tolist(),
                "coef": probe["coef"].tolist(),
                "intercept": float(probe["intercept"]),
            },
        },
        "ocr": {
            "countryCodes": sorted(ocr.COUNTRY_CODES),
            "minConfidence": 0.5,
            "countryMinConfidence": 0.6,
            "nameHeight": 96,
            "countryHeight": 120,
            # RapidOCR enlarges crops to 736 px before detection; name bars are already scaled to
            # ~100 px tall, and detecting at that scale keeps the browser fast.
            "det": {"thresh": 0.3, "boxThresh": 0.5, "unclipRatio": 1.6, "minSide": 160},
            "rec": {"height": 48, "minWidth": 320},
        },
    }
    (WEB_MODELS / "config.json").write_text(json.dumps(config))
    return config


def write_fixtures(dest: Path) -> None:
    """OpenCV reference outputs for the web port's image-operation tests."""
    import cv2

    rng = np.random.default_rng(0)
    dest.mkdir(parents=True, exist_ok=True)
    cases = []
    # A smooth image with texture: random pixels alone would hide interpolation differences.
    yy, xx = np.mgrid[0:97, 0:131]
    base = np.stack([(xx * 2) % 256, (yy * 3) % 256, ((xx + yy) * 5) % 256], -1).astype(np.uint8)
    noisy = np.clip(base.astype(int) + rng.integers(-30, 30, base.shape), 0, 255).astype(np.uint8)
    noisy.tofile(dest / "src.rgb")
    cv2.cvtColor(noisy, cv2.COLOR_RGB2LAB).tofile(dest / "lab.u8")
    for w, h in [(224, 224), (40, 30), (65, 49), (300, 210), (131, 50), (60, 140)]:
        out = cv2.resize(noisy, (w, h), interpolation=cv2.INTER_AREA)
        out.tofile(dest / f"area_{w}x{h}.rgb")
        cases.append([w, h])
    (dest / "meta.json").write_text(json.dumps({"width": 131, "height": 97, "area": cases}))
