"""Score the recognizer against known teams.

Ground truth comes from two places:
  * data/rosters/<event>.json - official rosters exported from Dracoviz (scripts/export_rosters.mjs);
  * labels/screens.json - which player each team grid in each screenshot belongs to, keyed by
    the screenshot path relative to the screenshots folder, e.g.
        {"EUIC 2026/image.webp": {"event": "2026-euic", "left": "PlayerOne", "right": "PlayerTwo"}}
    A single-team crop uses the key "team" instead of left/right. Teams of events that are not
    on Dracoviz yet can be written inline: {"team": {"player": "...", "pokemon": ["dunsparce",
    "ninetales_alolan_shadow", ...]}}.

Species are compared as sprite classes (cosmetic forms are indistinguishable on screen) and as a
multiset, since the overlay and the roster need not list a team in the same order.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .io import find_images, load_rgb
from .paths import DATA, LABELS
from .refs import load_species

SCREENS_FILE = LABELS / "screens.json"


def load_rosters() -> dict[str, dict[str, list[dict]]]:
    """event -> player name -> team [{speciesId, shadow}].

    Exported Dracoviz rosters live in data/rosters; hand-labeled teams for events that aren't on
    Dracoviz yet live in labels/rosters, in the same shape.
    """
    out = {}
    for path in sorted([*(DATA / "rosters").glob("*.json"), *(LABELS / "rosters").glob("*.json")]):
        out[path.stem] = {p["name"]: p["team"] for p in json.loads(path.read_text())}
    return out


def _sid_lookup():
    by_id = {s.speciesId: s.sid for s in load_species()}

    def sid(species_id: str) -> int | None:
        base = species_id.removesuffix("_shadow")
        if base in by_id:
            return by_id[base]
        # Roster ids sometimes carry a form suffix the gamemaster doesn't (or vice versa).
        parts = base.split("_")
        while len(parts) > 1:
            parts.pop()
            if "_".join(parts) in by_id:
                return by_id["_".join(parts)]
        return None

    return sid


def _truth_team(entry, event: str | None, rosters) -> tuple[str | None, list[tuple[int | None, bool]] | None]:
    sid = _sid_lookup()
    if isinstance(entry, dict):  # inline team
        team = [(sid(p), p.endswith("_shadow")) for p in entry["pokemon"]]
        return entry.get("player"), team
    roster = rosters.get(event or "", {}).get(entry)
    if roster is None:
        return entry, None
    return entry, [(sid(p["speciesId"]), bool(p["shadow"])) for p in roster]


def evaluate(screens_dir: Path, recognizer, verbose: bool = True) -> dict:
    labels = json.loads(SCREENS_FILE.read_text()) if SCREENS_FILE.exists() else {}
    rosters = load_rosters()
    sid = _sid_lookup()
    stats: dict[str, Counter] = defaultdict(Counter)
    mismatches = []

    for path in find_images(screens_dir):
        key = unicodedata.normalize("NFC", str(path.relative_to(screens_dir)))
        label = labels.get(key)
        if label is None:
            continue
        event = label.get("event")
        group = event or path.parent.name
        names = list(rosters.get(event or "", {}))
        teams = recognizer.recognize(load_rgb(path), roster_names=names)
        expected_sides = [s for s in ("left", "right", "team") if s in label]
        s = stats[group]
        s["screens"] += 1
        s["grids_expected"] += len(expected_sides)
        s["grids_found"] += min(len(teams), len(expected_sides))
        if len(teams) == 1 and expected_sides == ["team"]:
            pairs = [("team", teams[0])]
        else:
            pairs = [(t.side, t) for t in teams if t.side in expected_sides]
        for side, team in pairs:
            player, truth = _truth_team(label[side], event, rosters)
            if isinstance(label[side], str):
                s["names_total"] += 1
                s["names_ok"] += int(team.player == player)
            if truth is None:
                s["teams_unlabeled"] += 1
                continue
            predicted = [(sid(p["speciesId"]), p["shadow"]) for p in team.pokemon]
            truth_species = Counter(t[0] for t in truth)
            pred_species = Counter(p[0] for p in predicted)
            hit = sum((truth_species & pred_species).values())
            s["teams"] += 1
            s["species_total"] += len(truth)
            s["species_ok"] += hit
            truth_shadow = {t[0]: t[1] for t in truth}
            team_ok = hit == len(truth)
            for p_sid, p_shadow in predicted:
                if p_sid in truth_shadow:
                    s["shadow_total"] += 1
                    s["shadow_ok"] += int(p_shadow == truth_shadow[p_sid])
                    s["shadow_tp"] += int(p_shadow and truth_shadow[p_sid])
                    s["shadow_fp"] += int(p_shadow and not truth_shadow[p_sid])
                    s["shadow_fn"] += int(not p_shadow and truth_shadow[p_sid])
                    team_ok &= p_shadow == truth_shadow[p_sid]
            s["teams_exact"] += int(team_ok)
            if not team_ok:
                mismatches.append((key, side, player, team, truth))

    if verbose:
        _report(stats, mismatches)
    return {k: dict(v) for k, v in stats.items()}


def _pct(a, b) -> str:
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "    -"


def _report(stats, mismatches) -> None:
    species_by_sid = {}
    for s in load_species():
        species_by_sid.setdefault(s.sid, s.speciesId)
    print(f"{'event':<22}{'grids':>18}{'names':>18}{'species':>20}{'shadow':>20}{'teams':>16}")
    for group, s in sorted(stats.items()):
        print(
            f"{group:<22}{_pct(s['grids_found'], s['grids_expected']):>18}{_pct(s['names_ok'], s['names_total']):>18}"
            f"{_pct(s['species_ok'], s['species_total']):>20}{_pct(s['shadow_ok'], s['shadow_total']):>20}"
            f"{_pct(s['teams_exact'], s['teams']):>16}"
        )
        if s["shadow_total"]:
            print(f"{'':<22}shadow tp={s['shadow_tp']} fp={s['shadow_fp']} fn={s['shadow_fn']}")
    if mismatches:
        print("\nMismatches:")
    for key, side, player, team, truth in mismatches:
        print(f"  {key} [{side}] {player}")
        truth_names = [species_by_sid.get(t[0], "?") + ("*" if t[1] else "") for t in truth]
        print(f"    truth: {', '.join(sorted(truth_names))}")
        for p in team.pokemon:
            mark = "*" if p["shadow"] else ""
            alts = ", ".join(f"{n} {v:.2f}" for n, v in p["top3"])
            print(f"    pred:  {p['speciesId'].removesuffix('_shadow')}{mark} (violet={p['shadow_violet']} probe={p['shadow_probe']})  top3: {alts}")


def random_crop(img, grid, rng) -> "np.ndarray":
    """A human-style crop around one team: its grid and name bar plus random margins, rescaled
    and re-encoded."""
    import io as _io

    import cv2
    import numpy as np
    from PIL import Image

    H, W = img.shape[:2]
    x0, y0, x1, y1 = grid.bbox
    ny0 = grid.name_bar()[1]
    tw = grid.tile_w
    m = lambda: rng.uniform(0.0, 0.4) * tw  # noqa: E731
    cx0, cy0 = max(0, int(x0 - m())), max(0, int(ny0 - m()))
    cx1, cy1 = min(W, int(x1 + m())), min(H, int(y1 + rng.uniform(0.0, 0.25) * tw))
    crop = img[cy0:cy1, cx0:cx1]
    scale = rng.uniform(0.4, 1.5)
    scale = max(scale, 60 / tw)  # keep tiles at least 60px wide, as in the smallest real crops
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    buf = _io.BytesIO()
    Image.fromarray(crop).save(buf, "WEBP", quality=int(rng.uniform(40, 90)))
    return np.asarray(Image.open(buf).convert("RGB"))


def crop_suite(screens_dir: Path, recognizer, n: int = 5, seed: int = 0) -> dict:
    """Re-run every full-frame team on `n` random crops; predictions should not change."""
    import numpy as np

    from .grid import locate

    labels = json.loads(SCREENS_FILE.read_text()) if SCREENS_FILE.exists() else {}
    rng = np.random.default_rng(seed)
    stats: dict[str, Counter] = defaultdict(Counter)
    for path in find_images(screens_dir):
        key = unicodedata.normalize("NFC", str(path.relative_to(screens_dir)))
        label = labels.get(key)
        if not label or "left" not in label:
            continue
        img = load_rgb(path)
        full = {t.side: t for t in recognizer.recognize(img)}
        group = label.get("event") or path.parent.name
        for grid in locate(img):
            ref = full.get(grid.side)
            if ref is None:
                continue
            want = sorted(p["speciesId"] for p in ref.pokemon)
            for _ in range(n):
                teams = recognizer.recognize(random_crop(img, grid, rng))
                s = stats[group]
                s["crops"] += 1
                s["found"] += int(len(teams) == 1)
                got = sorted(p["speciesId"] for p in teams[0].pokemon) if teams else []
                s["same"] += int(got == want)
                if teams and got != want:
                    print(f"  {key} [{grid.side}] crop differs: {sorted(set(got) ^ set(want))}")
    for group, s in sorted(stats.items()):
        print(f"{group:<22} crops={s['crops']}  one grid found={_pct(s['found'], s['crops'])}  same team={_pct(s['same'], s['crops'])}")
    return {k: dict(v) for k, v in stats.items()}
