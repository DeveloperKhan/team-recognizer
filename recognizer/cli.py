"""Command line entry point: `uv run recognizer <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
                    print(f"    {p['speciesId']:<28} confidence={p['confidence']:.2f} violet={p['shadow_violet']:.2f} probe={p['shadow_probe']:.2f}")
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
    dataset.build(Path(args.screens).expanduser(), embedder, gallery, exclude_events=tuple(args.holdout))
    records, feats = dataset.load()
    events = sorted({r.event for r in records})
    print(f"{len(records)} labeled tiles from {', '.join(events)} ({sum(r.shadow for r in records)} shadow)")

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
    band_feats = np.vstack([feats["band"], embedder([shadow.band_only(v) for v in variants])]) if variants else feats["band"]
    index = np.concatenate([np.arange(len(records)), owners])
    sids = np.array([r.sid for r in records])[index]
    shadows = np.array([r.shadow for r in records])[index]

    shadow.fit(band_feats, shadows)
    print(f"shadow classifier ({len(shadows)} examples) -> {shadow.MODEL_FILE}")

    x = np.vstack([gallery.feats, tile_feats])
    y = np.concatenate([gallery.labels, sids])
    w = np.concatenate([np.ones(len(gallery.labels)), np.full(len(sids), args.real_weight)])
    head = train_head.train(x, y, gallery.classes, w)
    train_head.save(head, gallery.classes)
    print(f"species head -> {train_head.HEAD_FILE}")


def cmd_eval(args) -> None:
    from .evaluate import crop_suite, evaluate
    from .pipeline import Recognizer

    recognizer = Recognizer(read_names=not args.no_names)
    screens = Path(args.screens).expanduser()
    evaluate(screens, recognizer)
    if args.crops:
        print(f"\nCrop robustness ({args.crops} random crops per full-frame team):")
        recognizer.read_names = False
        crop_suite(screens, recognizer, n=args.crops)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recognizer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-refs", help="download sprites, harvest tile colors, build the gallery")
    p.add_argument("--screens", type=Path, help="screenshots to harvest overlay tile colors from")
    p.add_argument("--per-class", type=int, default=24, help="synthetic tiles per sprite class")
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
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="score against labels/screens.json and exported rosters")
    p.add_argument("--screens", default="~/Desktop/screenshots")
    p.add_argument("--no-names", action="store_true")
    p.add_argument("--crops", type=int, default=0, help="also test N random re-crops of each full-frame team")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
