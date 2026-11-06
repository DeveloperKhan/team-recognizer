"""Corrections shared from play-graphic-tool's Auto-Generate review: the teams people imported
from screenshots, with the species and shadow forms they confirmed or fixed by hand.

They arrive in the data folder (paths.DATA_DIR) and only train once reviewed:

    corrections/inbox/<id>/       as submitted: labels.json and one team-<n>.jpg per team
    corrections/accepted/<id>/    reviewed and accepted: these train
    corrections/rejected/<id>/

Each team-<n>.jpg is one team's panel (its name bar and six tiles, cut from the screenshot at
full resolution); labels.json gives each tile's box in it and its labels. The format is in the
README ("Your data").

Which tiles of an accepted submission train:
  * edited: changed by hand in the review, the cases the model got wrong (weight 2);
  * confident: left as recognized at confidence >= CONFIDENT (weight 1);
  * unchecked: left as recognized below that, which nobody may have looked at; only when the
    submission was accepted with --unchecked (weight 1).
Slots left unrecognized never train.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .io import load_rgb
from .paths import CACHE, DATA_DIR
from .pipeline import crop_tile

CORRECTIONS = DATA_DIR / "corrections"
STATES = ("inbox", "accepted", "rejected")
FORMAT = 1
CONFIDENT = 0.9
EDITED_WEIGHT = 2.0
REVIEW_FILE = CACHE / "corrections-review.html"


@dataclass
class CorrectionTile:
    submission: str
    team: int
    slot: int
    event: str
    species_id: str
    shadow: bool
    kind: str  # "edited", "confident" or "unchecked"
    weight: float
    tile: np.ndarray


def submissions(state: str) -> list[Path]:
    root = CORRECTIONS / state
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "labels.json").is_file())


def load_labels(folder: Path) -> dict:
    labels = json.loads((folder / "labels.json").read_text())
    if labels.get("format") != FORMAT:
        raise ValueError(f"{folder.name}: unsupported corrections format {labels.get('format')!r}")
    return labels


def event_slug(challonge_url: str | None) -> str:
    """The event a submission belongs to, named the way events are here:
    https://pokemongochampionshipseries.challonge.com/2027_GO_Baltimore -> 2027-baltimore."""
    path = urlparse(challonge_url or "").path.strip("/")
    slug = (path.split("/")[0] if path else "").lower()
    slug = re.sub(r"(^|_)go(_|$)", r"\1", slug)
    return re.sub(r"[^a-z0-9]+", "-", slug).strip("-") or "unknown"


def kind_of(slot: dict) -> str | None:
    if not slot.get("speciesId"):
        return None
    if slot.get("edited"):
        return "edited"
    return "confident" if slot["predicted"]["confidence"] >= CONFIDENT else "unchecked"


def _pixels(box, shape) -> tuple[float, float, float, float]:
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return x0 * w, y0 * h, x1 * w, y1 * h


def tiles(folder: Path, unchecked: bool | None = None) -> list[CorrectionTile]:
    """The tiles of one submission that train. `unchecked` defaults to what the review decided."""
    labels = load_labels(folder)
    if unchecked is None:
        unchecked = bool(labels.get("review", {}).get("unchecked", False))
    event = event_slug(labels.get("event", {}).get("challongeUrl"))
    out = []
    for t, team in enumerate(labels["teams"]):
        panel = load_rgb(folder / team["image"])
        for s, slot in enumerate(team["slots"]):
            kind = kind_of(slot)
            if kind is None or (kind == "unchecked" and not unchecked):
                continue
            out.append(
                CorrectionTile(
                    submission=folder.name,
                    team=t,
                    slot=s,
                    event=event,
                    species_id=slot["speciesId"],
                    shadow=bool(slot["shadow"]),
                    kind=kind,
                    weight=EDITED_WEIGHT if kind == "edited" else 1.0,
                    tile=crop_tile(panel, _pixels(slot["box"], panel.shape)),
                )
            )
    return out


def _fingerprint(tile: np.ndarray) -> str:
    """Survives re-encoding: the same panel shared twice gives the same fingerprint."""
    small = cv2.resize(cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
    return hashlib.sha1((small // 16).tobytes()).hexdigest()


def training_tiles(exclude_events: tuple[str, ...] = ()) -> list[CorrectionTile]:
    """The accepted corrections' training tiles, each look of each label once."""
    seen: set[tuple[str, str, bool]] = set()
    out = []
    for folder in submissions("accepted"):
        for tile in tiles(folder):
            key = (_fingerprint(tile.tile), tile.species_id, tile.shadow)
            if tile.event in exclude_events or key in seen:
                continue
            seen.add(key)
            out.append(tile)
    return out


def find(submission_id: str) -> tuple[str, Path]:
    for state in STATES:
        folder = CORRECTIONS / state / submission_id
        if (folder / "labels.json").is_file():
            return state, folder
    raise SystemExit(f"no correction submission {submission_id!r} in {CORRECTIONS}")


def move(submission_id: str, state: str, unchecked: bool = False) -> Path:
    """Accept or reject a submission, recording the decision in its labels.json."""
    _, folder = find(submission_id)
    labels = load_labels(folder)
    labels["review"] = {
        "state": state,
        "unchecked": bool(unchecked) if state == "accepted" else False,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    dest = CORRECTIONS / state / submission_id
    if dest != folder:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(dest))
    return dest


def summary() -> dict[str, dict[str, int]]:
    """Per state: submissions, teams, and tiles of each kind."""
    out = {}
    for state in STATES:
        counts = {"submissions": 0, "teams": 0, "edited": 0, "confident": 0, "unchecked": 0}
        for folder in submissions(state):
            labels = load_labels(folder)
            counts["submissions"] += 1
            counts["teams"] += len(labels["teams"])
            for team in labels["teams"]:
                for slot in team["slots"]:
                    kind = kind_of(slot)
                    if kind:
                        counts[kind] += 1
        out[state] = counts
    return out


# --- review ---------------------------------------------------------------------------------


def _data_uri(img: np.ndarray, height: int = 96) -> str:
    if img.shape[0] > height:
        img = cv2.resize(
            img, (max(1, round(img.shape[1] * height / img.shape[0])), height), interpolation=cv2.INTER_AREA
        )
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode() if ok else ""


def _label(species_id: str, shadow: bool) -> str:
    return (species_id or "?") + (" (shadow)" if shadow else "")


def review(state: str = "inbox", out: Path = REVIEW_FILE) -> Path:
    """An HTML contact sheet of a state's submissions: each team's name bar, then every tile
    above what it was recognized as and what it was kept or corrected to."""
    esc = html.escape
    parts = []
    for folder in submissions(state):
        labels = load_labels(folder)
        teams = labels["teams"]
        edited = sum(1 for t in teams for s in t["slots"] if s.get("edited"))
        event = labels.get("event", {})
        parts.append(
            f'<section><h2>{esc(folder.name)}</h2><p class="meta">{esc(event.get("name") or "")} '
            f"· {esc(labels.get('via', ''))} · {len(teams)} teams · {edited} edited tiles · shared "
            f"{esc(labels.get('receivedAt', labels.get('createdAt', '')))}</p>"
            f"<code>uv run recognizer corrections accept {esc(folder.name)}</code> "
            f"<code>... --unchecked</code> <code>uv run recognizer corrections reject {esc(folder.name)}</code>"
        )
        for team in teams:
            panel = load_rgb(folder / team["image"])
            match = team.get("nameMatch")
            how = "picked by hand" if match == "manual" else f"via {match.get('via')}, {match.get('edits')} edits"
            name_bar = (
                f'<img class="name" src="{_data_uri(crop_tile(panel, _pixels(team["nameBox"], panel.shape)), 40)}">'
                if team.get("nameBox")
                else ""
            )
            player = team.get("player", {})
            parts.append(
                f'<div class="team"><div class="head">{name_bar} read <b>{esc(team.get("read") or "?")}</b> '
                f"→ #{player.get('rank')} <b>{esc(player.get('name', ''))}</b> ({esc(how)}) "
                f"· model {esc(str(team.get('modelVersion')))} · {esc(team.get('source', ''))}</div><div class='tiles'>"
            )
            for slot in team["slots"]:
                kind = kind_of(slot) or "empty"
                tile = crop_tile(panel, _pixels(slot["box"], panel.shape))
                predicted = slot["predicted"]
                was = (
                    f"<span class='was'>was {esc(_label(predicted['speciesId'], predicted['shadow']))}</span>"
                    if slot.get("edited")
                    else ""
                )
                parts.append(
                    f"<figure class='{kind}'><img src='{_data_uri(tile)}'><figcaption>"
                    f"<b>{esc(_label(slot['speciesId'], slot['shadow']))}</b>{was}"
                    f"<span>{kind} · {round(predicted['confidence'] * 100)}%</span></figcaption></figure>"
                )
            parts.append("</div></div>")
        parts.append("</section>")

    body = "".join(parts) or f"<p>No submissions in {esc(state)}.</p>"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "<!doctype html><meta charset='utf-8'><title>Corrections review</title><style>"
        "body{font:14px system-ui,sans-serif;margin:24px;color:#16181d}section{margin-bottom:32px}"
        "h2{font-size:16px;margin:0}.meta{color:#5f6672;margin:4px 0 8px}code{background:#f1f2f4;padding:2px 5px;"
        "border-radius:4px;font-size:12px}.team{border:1px solid #e2e5ea;border-radius:10px;padding:10px;margin:10px 0}"
        ".head{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.name{height:28px;border-radius:4px}"
        ".tiles{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}figure{margin:0;width:104px;text-align:center;"
        "padding:4px;border-radius:8px;border:2px solid transparent}figure img{width:96px;height:96px;object-fit:contain;"
        "background:#000;border-radius:6px}figcaption{font-size:12px;display:flex;flex-direction:column}"
        "figcaption span{color:#5f6672}.edited{border-color:#2563eb;background:#e8efff}.was{color:#b91c1c!important}"
        ".unchecked{border-color:#f59e0b;background:#fff7e6}.empty{opacity:.5}"
        f"</style><h1>Corrections: {esc(state)}</h1><p>Blue: changed by hand (trains). Amber: left as "
        "recognized below 90% (trains only with <code>--unchecked</code>). The rest: left as recognized at 90%+ "
        "(trains).</p>" + body
    )
    return out


# --- scoring ---------------------------------------------------------------------------------


def score(recognizer, events: list[str] | None = None) -> dict[str, dict[str, int]]:
    """How the current model reads the accepted corrections' panels, per event: species and
    shadow per tile, overall and on the tiles people changed by hand."""
    from .evaluate import _sid_lookup

    sid = _sid_lookup()
    stats: dict[str, dict[str, int]] = {}
    for folder in submissions("accepted"):
        labels = load_labels(folder)
        event = event_slug(labels.get("event", {}).get("challongeUrl"))
        if events and event not in events:
            continue
        s = stats.setdefault(
            event, {"teams": 0, "no grid": 0, "tiles": 0, "species": 0, "shadow": 0, "edited": 0, "edited ok": 0}
        )
        for team in labels["teams"]:
            s["teams"] += 1
            found = recognizer.recognize(load_rgb(folder / team["image"]))
            if len(found) != 1:
                s["no grid"] += 1
                continue
            for slot, guess in zip(team["slots"], found[0].pokemon):
                if not slot["speciesId"]:
                    continue
                species_ok = sid(guess["speciesId"]) == sid(slot["speciesId"])
                shadow_ok = bool(guess["shadow"]) == bool(slot["shadow"])
                s["tiles"] += 1
                s["species"] += species_ok
                s["shadow"] += shadow_ok
                if slot.get("edited"):
                    s["edited"] += 1
                    s["edited ok"] += species_ok and shadow_ok
    return stats
