"""Pokémon GO's own art for each sprite class.

The overlay draws each Pokémon as it looks in Pokémon GO, in the player's form, costume and
shininess. Pokémon Showdown's sprites and HOME renders often differ in pose and style (HOME shows
Stunfisk face-on, the overlay side-on), and lack GO-only looks like the white-flowered Florges.
So GO's icon art is the main reference for synthetic tiles, where it exists.

Source: PokeMiners/pogo_assets, mined from the game, pinned to a commit. The images are
downloaded into the cache at build time and never redistributed.

Each icon is named pm<dex>[.f<FORM>][.c<COSTUME>][.g<GENDER>][.s].icon.png (.s: shiny). Within a
dex number, an icon goes to the sprite class whose speciesId its form matches (GALARIAN ->
stunfisk_galarian, PALDEA_COMBAT -> tauros_combat). Icons of forms that aren't classes here (Mega,
Gigantamax, battle-only forms) are skipped, and everything else (flower colors, costumes,
genders, shinies) is cosmetic and goes to the dex's default class.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np
from PIL import Image

from .paths import CACHE

GO_REPO = "PokeMiners/pogo_assets"
GO_COMMIT = "db10df75d6f6f92018d9786af0aecc238fecc4ef"  # 2026-10-28
GO_DIR = "Images/Pokemon/Addressable Assets"
RAW_URL = f"https://raw.githubusercontent.com/{GO_REPO}/{GO_COMMIT}/{quote(GO_DIR)}/{{name}}"
GO_REFS = CACHE / "refs" / "go"
INDEX = GO_REFS / "index.json"
MAX_PER_CLASS = 12

# Forms that are never on an overlay team, so their art would only mislead
SKIP_FORMS = {
    "MEGA",
    "MEGA_X",
    "MEGA_Y",
    "MEGA_Z",
    "PRIMAL",
    "GIGANTAMAX",
    "ETERNAMAX",
    "ZEN",
    "GALARIAN_ZEN",
    "HANGRY",
}
# GO's spelling of the regional tokens in PvPoke's speciesIds
ALIASES = {"alola": "alolan", "paldea": "paldean"}

_ICON = re.compile(r"^pm(\d+)((?:\.[fcg][A-Za-z0-9_]+)*)(\.s)?\.icon\.png$")


@dataclass(frozen=True)
class Icon:
    name: str
    dex: int
    form: str  # "" for the default form
    costume: str
    gender: str
    shiny: bool


def parse(name: str) -> Icon | None:
    m = _ICON.match(name)
    if not m:
        return None
    parts = {p[0]: p[1:] for p in m.group(2).split(".")[1:]}
    return Icon(name, int(m.group(1)), parts.get("f", ""), parts.get("c", ""), parts.get("g", ""), bool(m.group(3)))


def _tokens(text: str) -> set[str]:
    return {ALIASES.get(t, t) for t in text.lower().split("_") if t}


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _match_score(form: str, species_ids: list[str]) -> int:
    """How well a GO form names a class: shared words, plus a bonus when the form's letters
    appear whole in a speciesId (POMPOM in oricorio_pom_pom)."""
    words = _tokens(form)
    best = 0
    for sid in species_ids:
        score = len(words & _tokens(sid)) + (2 if _alnum(form) and _alnum(form) in _alnum(sid) else 0)
        best = max(best, score)
    return best


def _priority(icon: Icon) -> tuple:
    """Plain art first, then shiny, then cosmetic forms and genders, then costumes."""
    return (bool(icon.costume), bool(icon.form), bool(icon.gender), icon.shiny, icon.name)


def assign(icons: list[Icon], classes: dict[int, list]) -> tuple[dict[int, list[Icon]], dict[str, int]]:
    """Icons for each sprite class (`classes`: sid -> its Species), and counts of what was
    skipped or sent to a default class, for checking the mapping."""
    by_dex: dict[int, list[tuple[int, list[str]]]] = {}
    for sid, members in classes.items():
        by_dex.setdefault(members[0].dex, []).append((sid, [m.speciesId for m in members]))
    stats = {"matched": 0, "default": 0, "skipped": 0, "no class": 0}
    out: dict[int, list[Icon]] = {}
    for icon in icons:
        candidates = by_dex.get(icon.dex)
        if not candidates:
            stats["no class"] += 1
            continue
        if icon.form in SKIP_FORMS:
            stats["skipped"] += 1
            continue
        scored = [(_match_score(icon.form, ids), -len(min(ids, key=len)), -sid, sid) for sid, ids in candidates]
        best = max(scored)
        if icon.form and best[0] > 0:
            stats["matched"] += 1
            target = best[3]
        else:
            # The default form, or a cosmetic one: the class with the plainest speciesId
            stats["default"] += 1
            target = min(candidates, key=lambda c: (len(_tokens(min(c[1], key=len))), c[0]))[0]
        out.setdefault(target, []).append(icon)
    return {sid: sorted(found, key=_priority)[:MAX_PER_CLASS] for sid, found in out.items()}, stats


def list_icons() -> list[Icon]:
    """The icon names at the pinned commit, from GitHub's API (cached in INDEX)."""
    if INDEX.exists():
        names = json.loads(INDEX.read_text())
    else:
        import requests

        api = f"https://api.github.com/repos/{GO_REPO}"
        parent = requests.get(f"{api}/contents/{quote(GO_DIR.rsplit('/', 1)[0])}?ref={GO_COMMIT}", timeout=60)
        parent.raise_for_status()
        tree_sha = next(e["sha"] for e in parent.json() if e["name"] == GO_DIR.rsplit("/", 1)[1])
        tree = requests.get(f"{api}/git/trees/{tree_sha}", timeout=60)
        tree.raise_for_status()
        names = sorted(e["path"] for e in tree.json()["tree"] if e["path"].endswith(".icon.png"))
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(json.dumps(names))
    return [icon for icon in (parse(n) for n in names) if icon]


def download(classes: dict[int, list], workers: int = 8) -> dict[str, int]:
    """Download each class's icons into GO_REFS/<sid>/. Returns the mapping's counts."""
    import requests
    from tqdm import tqdm

    assigned, stats = assign(list_icons(), classes)
    session = requests.Session()
    jobs = [(icon.name, GO_REFS / str(sid) / icon.name) for sid, icons in assigned.items() for icon in icons]

    def fetch(job) -> bool:
        name, dest = job
        if dest.exists() and dest.stat().st_size:
            return True
        for _ in range(3):
            try:
                response = session.get(RAW_URL.format(name=quote(name)), timeout=30)
            except requests.RequestException:
                continue  # GitHub's raw host times out now and then
            if response.status_code != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(response.content)
            return True
        return False

    with ThreadPoolExecutor(workers) as pool:
        ok = list(tqdm(pool.map(fetch, jobs), total=len(jobs), desc="GO art", mininterval=5))
    return {**stats, "classes": len(assigned), "icons": len(jobs), "failed": ok.count(False)}


def reference_images(sid: int) -> list[np.ndarray]:
    """A class's GO icons as RGBA arrays, in priority order (see _priority)."""
    folder = GO_REFS / str(sid)
    if not folder.is_dir():
        return []
    icons = sorted((parse(p.name) for p in folder.glob("*.icon.png")), key=lambda i: _priority(i) if i else ())
    return [np.asarray(Image.open(folder / icon.name).convert("RGBA")) for icon in icons if icon]
