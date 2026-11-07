"""Command line entry point: `uv run recognizer <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_update_species(args) -> None:
    from .species import update

    species = update()
    print(f"data/species.json: {len(species)} species, {len({s['sid'] for s in species})} sprites")


def cmd_build_refs(args) -> None:
    from .embed import Embedder, build_gallery
    from .io import find_images, load_rgb
    from .refs import download_all
    from .synth import harvest_palette

    download_all()
    if args.screens:
        palette = harvest_palette([load_rgb(p) for p in find_images(args.screens)])
        print(f"Palette: {len(palette)} tile fills")
    build_gallery(Embedder(), per_class=args.per_class)


def cmd_recognize(args) -> None:
    from .evaluate import load_rosters
    from .io import find_images, load_rgb
    from .pipeline import Recognizer

    names = list(load_rosters().get(args.event, {})) if args.event else None
    recognizer = Recognizer(read_names=not args.no_names)
    results = {}
    for path in [p for arg in args.images for p in find_images(arg)]:
        teams = recognizer.recognize(load_rgb(path), roster_names=names)
        results[str(path)] = [t.to_dict() for t in teams]
        if not args.json:
            print(path)
            if not teams:
                print("  no team grid found")
            for t in teams:
                print(f"  [{t.side or 'team'}] {t.player or '?'}  (read: {t.player_raw!r})")
                for p in t.pokemon:
                    print(
                        f"    {p['speciesId']:<28} confidence={p['confidence']:.2f} violet={p['shadow_violet']:.2f} probe={p['shadow_probe']:.2f}"
                    )
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()


def cmd_label_screens(args) -> None:
    """Propose labels/screens.json entries: each grid's player, read by OCR and matched against
    the rosters (the folder's event first, then every other exported event)."""
    import unicodedata

    from . import ocr
    from .evaluate import SCREENS_FILE, load_rosters
    from .grid import locate
    from .io import find_images, load_rgb
    from .paths import LABELS

    events = json.loads((LABELS / "events.json").read_text())
    rosters = load_rosters()
    labels = json.loads(SCREENS_FILE.read_text()) if SCREENS_FILE.exists() else {}
    root = Path(args.screens).expanduser()
    for path in find_images(root):
        key = unicodedata.normalize("NFC", str(path.relative_to(root)))
        if key in labels and not args.overwrite:
            continue
        folder_event = events.get(unicodedata.normalize("NFC", path.parent.name))
        if folder_event is None:
            continue  # folder not mapped to an event in labels/events.json
        img = load_rgb(path)
        grids = locate(img)
        if not grids:
            print(f"{key}: no grid")
            continue
        entry: dict = {"event": folder_event}
        for grid in grids:
            raw = ocr.read_name(img, grid.name_bar())
            side = grid.side or "team"
            # Look in other events' rosters only when this event's roster exists and lacks the
            # name (a screenshot filed under the wrong event). Players attend several events and
            # their teams change, so an event without a roster must not borrow another's.
            order = [folder_event]
            if folder_event in rosters:
                order += [e for e in rosters if e != folder_event]
            for event in order:
                name, _ = ocr.match_name(raw, list(rosters.get(event, {})))
                if name:
                    entry[side] = name
                    entry["event"] = event
                    break
            else:
                entry[side] = raw
                print(f"{key} [{side}]: {raw!r} not in any exported roster")
        labels[key] = entry
        print(key, entry)
    SCREENS_FILE.write_text(json.dumps(dict(sorted(labels.items())), indent=2, ensure_ascii=False) + "\n")


def cmd_train(args) -> None:
    import numpy as np

    from . import dataset, shadow, train_head
    from .embed import Embedder, Gallery
    from .synth import degrade

    embedder = Embedder()
    gallery = Gallery()
    dataset.build(
        Path(args.screens).expanduser(),
        embedder,
        gallery,
        exclude_events=tuple(args.holdout),
        use_corrections=not args.no_corrections,
    )
    records, feats = dataset.load()
    events = sorted({r.event for r in records})
    shared = [r for r in records if r.source == "correction"]
    print(f"{len(records)} labeled tiles from {', '.join(events)} ({sum(r.shadow for r in records)} shadow)")
    if shared:
        print(f"  of which {len(shared)} from shared corrections ({sum(r.weight > 1 for r in shared)} changed by hand)")

    # Screenshots arrive at every size and compression level, so each real tile is also seen
    # degraded the way the synthetic ones are.
    rng = np.random.default_rng(0)
    variants, owners = [], []
    for i, record in enumerate(records):
        tile = dataset.image(record)
        for _ in range(args.augment):
            variants.append(degrade(tile, rng))
            owners.append(i)
    owners = np.array(owners, int)
    tile_feats = np.vstack([feats["tile"], embedder(variants)]) if variants else feats["tile"]
    band_feats = (
        np.vstack([feats["band"], embedder([shadow.band_only(v) for v in variants])]) if variants else feats["band"]
    )
    index = np.concatenate([np.arange(len(records)), owners])
    sids = np.array([r.sid for r in records])[index]
    shadows = np.array([r.shadow for r in records])[index]

    shadow.fit(band_feats, shadows)
    print(f"shadow classifier ({len(shadows)} examples) -> {shadow.MODEL_FILE}")

    x = np.vstack([gallery.feats, tile_feats])
    y = np.concatenate([gallery.labels, sids])
    # Real tiles outweigh synthetic ones; corrections changed by hand count extra
    real_weights = args.real_weight * np.array([r.weight for r in records])[index]
    w = np.concatenate([np.ones(len(gallery.labels)), real_weights])
    head = train_head.train(x, y, gallery.classes, w)
    train_head.save(head, gallery.classes)
    print(f"species head -> {train_head.HEAD_FILE}")


def cmd_export_onnx(args) -> None:
    """Export DINOv2 to ONNX for PyTorch-free inference: full precision to the cache, then int8
    weights (a quarter of the size, same accuracy on held-out events) to models/."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from . import dataset
    from .embed import ONNX_FILE, OnnxEmbedder, TorchEmbedder
    from .paths import CACHE

    full = CACHE / "dinov2_vits14.fp32.onnx"
    torch_embedder = TorchEmbedder()
    torch_embedder.export_onnx(full)
    quantize_dynamic(str(full), str(ONNX_FILE), weight_type=QuantType.QInt8)
    records, _ = dataset.load()
    tiles = [dataset.image(r) for r in records[:64]]
    reference = torch_embedder(tiles)
    for path in (full, ONNX_FILE):
        cos = (reference * OnnxEmbedder(path)(tiles)).sum(1)
        print(
            f"{path.name} ({path.stat().st_size / 1e6:.0f} MB): cosine to torch features min={cos.min():.4f} mean={cos.mean():.4f}"
        )


def cmd_export_web(args) -> None:
    from . import web_export

    config = web_export.export()
    web_export.write_fixtures(web_export.FIXTURES)
    sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(web_export.WEB_MODELS.iterdir())}
    print(f"web/models ({config['version']}):", sizes)


def cmd_write_fixtures(args) -> None:
    from . import web_export

    web_export.write_fixtures(web_export.FIXTURES)
    print(f"wrote {web_export.FIXTURES.relative_to(web_export.ROOT)}")


def cmd_smoke(args) -> None:
    import cv2
    import numpy as np

    from .io import load_rgb
    from .paths import SMOKE
    from .synth import extract_smoke

    SMOKE.mkdir(parents=True, exist_ok=True)
    sources = [(Path(image).stem.replace(" ", "_"), load_rgb(image)) for image in args.images]
    if args.from_tiles:
        from . import dataset

        def purple_body(tile) -> bool:
            # The smoke is told apart by color, so a purple Pokémon would pass for smoke
            h, w = tile.shape[:2]
            hsv = cv2.cvtColor(tile[int(0.3 * h) : int(0.75 * h), int(0.3 * w) : int(0.7 * w)], cv2.COLOR_RGB2HSV)
            hue, sat = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
            return float(np.mean((hue >= 116) & (hue <= 165) & (sat >= 50))) > 0.04

        records, _ = dataset.load()
        for r in records:
            tile = dataset.image(r)
            if r.shadow and r.source == "screens" and not purple_body(tile):
                sources.append((f"tile_{Path(r.file).stem}", tile))
    kept = 0
    for name, img in sources:
        rgba = extract_smoke(img)
        cover = float((rgba[..., 3] > 128).mean())
        if cover < 0.04:
            continue  # too faint to teach anything (dimmed or fainted shadows)
        cv2.imwrite(str(SMOKE / f"{name}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        kept += 1
    print(f"{kept} smoke textures from {len(sources)} images in {SMOKE}")
    print("Rebuild the gallery (build-refs) to use them; commit the data folder to keep them.")


def cmd_corrections(args) -> None:
    from . import corrections

    if args.action == "list":
        for state, counts in corrections.summary().items():
            print(
                f"{state:<9} {counts['submissions']:>4} submissions  {counts['teams']:>4} teams  "
                f"{counts['edited']:>4} edited  {counts['confident']:>4} confident  {counts['unchecked']:>4} unchecked tiles"
            )
        print(f"(in {corrections.CORRECTIONS})")
    elif args.action == "review":
        print(f"wrote {corrections.review(args.state)}")
    elif args.action in ("accept", "reject"):
        state = "accepted" if args.action == "accept" else "rejected"
        ids = args.ids or ([f.name for f in corrections.submissions("inbox")] if getattr(args, "all", False) else [])
        if not ids:
            raise SystemExit("name the submissions to move, or pass --all for everything in the inbox")
        for submission_id in ids:
            dest = corrections.move(
                submission_id,
                state,
                unchecked=getattr(args, "unchecked", False),
                skip=getattr(args, "skip", None),
            )
            print(f"{submission_id} -> {dest.parent.name}")
        print("Commit the data folder to keep the decision.")


def cmd_eval(args) -> None:
    from .evaluate import crop_suite, evaluate
    from .pipeline import Recognizer

    if args.corrections is not None:
        from . import corrections

        stats = corrections.score(Recognizer(read_names=False), args.corrections or None)
        pct = lambda a, b: f"{100 * a / b:.1f}% ({a}/{b})" if b else "-"  # noqa: E731
        print(f"{'event':<22} {'teams':>6} {'no grid':>8} {'species':>18} {'shadow':>18} {'edited tiles right':>20}")
        for event, s in sorted(stats.items()):
            print(
                f"{event:<22} {s['teams']:>6} {s['no grid']:>8} {pct(s['species'], s['tiles']):>18} "
                f"{pct(s['shadow'], s['tiles']):>18} {pct(s['edited ok'], s['edited']):>20}"
            )
        if not stats:
            print("No accepted corrections to score.")
        return

    screens = Path(args.screens).expanduser()
    if args.predictions:
        evaluate(screens, None, predictions=json.loads(Path(args.predictions).read_text()))
        return
    recognizer = Recognizer(read_names=not args.no_names)
    evaluate(screens, recognizer)
    if args.crops:
        print(f"\nCrop robustness ({args.crops} random crops per full-frame team):")
        recognizer.read_names = False
        crop_suite(screens, recognizer, n=args.crops)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recognizer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("update-species", help="rebuild data/species.json from PvPoke, Showdown and Smogon")
    p.set_defaults(func=cmd_update_species)

    p = sub.add_parser("build-refs", help="download sprites, harvest tile colors, build the gallery")
    p.add_argument("--screens", type=Path, help="screenshots to harvest overlay tile colors from")
    p.add_argument("--per-class", type=int, default=48, help="synthetic tiles per sprite class")
    p.set_defaults(func=cmd_build_refs)

    p = sub.add_parser("recognize", help="identify the teams in screenshots")
    p.add_argument("images", nargs="+", help="image files or folders")
    p.add_argument("--event", help="Dracoviz event slug (e.g. 2027-nice) to match player names against")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-names", action="store_true", help="skip OCR of player names")
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("label-screens", help="propose which player each grid belongs to (OCR + rosters)")
    p.add_argument("--screens", default="~/Desktop/screenshots")
    p.add_argument("--overwrite", action="store_true", help="redo screenshots that already have labels")
    p.set_defaults(func=cmd_label_screens)

    p = sub.add_parser("train", help="fit the shadow classifier and species head on labeled screenshots")
    p.add_argument("--screens", default="~/Desktop/screenshots")
    p.add_argument("--holdout", nargs="*", default=[], help="events to leave out (for honest evaluation)")
    p.add_argument("--real-weight", type=float, default=5.0, help="loss weight of real tiles vs synthetic")
    p.add_argument("--augment", type=int, default=8, help="degraded copies of each real tile")
    p.add_argument("--no-corrections", action="store_true", help="leave out accepted shared corrections")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export-onnx", help="export the image model to ONNX for PyTorch-free inference")
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("export-web", help="write the models and settings the browser build uses (web/models)")
    p.set_defaults(func=cmd_export_web)

    p = sub.add_parser("write-fixtures", help="write the web build's image-operation test fixtures (no PyTorch needed)")
    p.set_defaults(func=cmd_write_fixtures)

    p = sub.add_parser("eval", help="score against labels/screens.json and exported rosters")
    p.add_argument("--screens", default="~/Desktop/screenshots")
    p.add_argument("--no-names", action="store_true")
    p.add_argument("--crops", type=int, default=0, help="also test N random re-crops of each full-frame team")
    p.add_argument("--predictions", help="score predictions from a file (e.g. the web build's parity run) instead")
    p.add_argument(
        "--corrections",
        nargs="*",
        metavar="EVENT",
        help="score accepted shared corrections instead (optionally only these events)",
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("smoke", help="cut shadow smoke textures from screenshots of shadow tiles")
    p.add_argument("images", nargs="*", help="screenshots, each of one shadow tile")
    p.add_argument("--from-tiles", action="store_true", help="also cut them from the labeled shadow tiles")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("corrections", help="review corrections shared from play-graphic-tool")
    actions = p.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="count submissions and tiles in the inbox, accepted and rejected")
    a = actions.add_parser("review", help="write an HTML contact sheet of a state's submissions")
    a.add_argument("--state", choices=["inbox", "accepted", "rejected"], default="inbox")
    a = actions.add_parser("accept", help="accept submissions so they train")
    a.add_argument("ids", nargs="*")
    a.add_argument("--all", action="store_true", help="everything in the inbox")
    a.add_argument("--unchecked", action="store_true", help="also train on unsure tiles left as recognized")
    a.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="TEAM:TILE",
        help="tiles to leave out (the numbers under each tile in the review sheet), e.g. 8:4",
    )
    a = actions.add_parser("reject", help="reject submissions")
    a.add_argument("ids", nargs="+")
    p.set_defaults(func=cmd_corrections)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
