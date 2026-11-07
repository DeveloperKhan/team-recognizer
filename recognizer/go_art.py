"""Pokémon GO's own art for each sprite class.

The overlay draws each Pokémon as it looks in Pokémon GO, in the player's form, costume and
shininess. Pokémon Showdown's sprites and HOME renders often differ in pose and style (HOME shows
Stunfisk face-on, the overlay side-on), and lack GO-only looks like the white-flowered Florges.
So GO's art is half of each class's references for synthetic tiles, where it exists.

Two sources, both downloaded into the cache at build time and never redistributed:
  * the Pokémon GO wiki on Fandom (Category:Pokémon_sprites), preferred: it covers newer species
    the mined assets lack (Cramorant, Mimikyu) and frames each Pokémon like the overlay. Files are
    named "<Name>[ <form words>][ female][ shiny].png";
  * PokeMiners/pogo_assets, mined from the game and pinned to a commit. Icons are named
    pm<dex>[.f<FORM>][.c<COSTUME>][.g<GENDER>][.s].icon.png (.s: shiny).

Within a dex number, an image goes to the sprite class whose speciesId its form matches (GALARIAN
-> stunfisk_galarian, PALDEA_COMBAT -> tauros_combat, a female look -> indeedee_female). Forms that
are never on a team (Mega, Gigantamax, battle-only forms) are skipped, and everything else (flower
colors, costumes, genders, shinies) is cosmetic and goes to the dex's default class.
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from urllib.parse import quote, urlencode

import numpy as np
from PIL import Image

from .paths import CACHE

GO_REPO = "PokeMiners/pogo_assets"
GO_COMMIT = "db10df75d6f6f92018d9786af0aecc238fecc4ef"  # 2026-10-28
GO_DIR = "Images/Pokemon/Addressable Assets"
RAW_URL = f"https://raw.githubusercontent.com/{GO_REPO}/{GO_COMMIT}/{quote(GO_DIR)}/{{name}}"
FANDOM_API = "https://pokemongo.fandom.com/api.php"
FANDOM_CATEGORY = "Category:Pokémon_sprites"
USER_AGENT = "team-recognizer (https://github.com/DeveloperKhan/team-recognizer)"

GO_REFS = CACHE / "refs" / "go"
INDEX = GO_REFS / "index.json"  # PokeMiners icon names at GO_COMMIT
FANDOM_INDEX = GO_REFS / "fandom-index.json"  # Fandom file titles and urls
ORDER = "order.json"  # in each class folder: the images chosen for it, best first
MAX_PER_CLASS = 12

# Words of forms that are never on an overlay team, so their art would only mislead
SKIP_WORDS = {"mega", "primal", "gigantamax", "eternamax", "zen", "hangry"}
# Spellings in the sources that differ from PvPoke's speciesIds, applied to a form before matching
FORM_ALIASES = [
    ("COMPLETE_TEN_PERCENT", "COMPLETE"),
    ("COMPLETE_FIFTY_PERCENT", "COMPLETE"),
    ("TEN_PERCENT", "10"),
    ("10PERCENT", "10"),
    ("FIFTY_PERCENT", ""),  # Zygarde's default form
]
WORD_ALIASES = {"alola": "alolan", "paldea": "paldean"}

_ICON = re.compile(r"^pm(\d+)((?:\.[fcg][A-Za-z0-9_]+)*)(\.s)?\.icon\.png$")


@dataclass(frozen=True)
class Icon:
    name: str  # the file name it's saved under
    dex: int
    form: str  # "" for the default form
    costume: str
    gender: str
    shiny: bool
    source: str = "pogo"  # "pogo" or "fandom"
    url: str = ""
    kind: str = ""  # set by assign(): see KIND_WEIGHTS


def parse(name: str) -> Icon | None:
    """A PokeMiners icon from its file name."""
    m = _ICON.match(name)
    if not m:
        return None
    parts = {p[0]: p[1:] for p in m.group(2).split(".")[1:]}
    return Icon(name, int(m.group(1)), parts.get("f", ""), parts.get("c", ""), parts.get("g", ""), bool(m.group(3)))


def _fold(text: str) -> str:
    """Lowercase words, symbols spelled out, accents and punctuation dropped."""
    text = text.replace("♀", " female").replace("♂", " male")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", "", text)


def species_dex(species: list) -> dict[str, int]:
    """Folded base names ("stunfisk", "nidoran female", "mr mime") -> dex number."""
    out = {}
    for s in species:
        base = re.sub(r"\s*\(.*\)$", "", s.speciesName).split("_")[0]
        out.setdefault(_fold(base).strip(), s.dex)
    return out


def parse_fandom(title: str, url: str, names: dict[str, int]) -> Icon | None:
    """A Fandom sprite from its title, e.g. "Stunfisk galarian shiny.png"."""
    if not title.lower().endswith(".png"):
        return None  # the category also holds a few cries (.ogg)
    words = _fold(re.sub(r"\.png$", "", title, flags=re.I)).split()
    for k in range(min(4, len(words)), 0, -1):  # the longest prefix that names a species
        dex = names.get(" ".join(words[:k]))
        if dex is not None:
            rest = words[k:]
            shiny = "shiny" in rest
            gender = "female" if "female" in rest else ""
            form = "_".join(w for w in rest if w not in ("shiny", "female")).upper()
            safe = re.sub(r"[^A-Za-z0-9]+", "_", re.sub(r"\.png$", "", title, flags=re.I)).strip("_")
            return Icon(f"fandom_{safe}.png", dex, form, "", gender, shiny, "fandom", url)
    return None


def _tokens(text: str) -> set[str]:
    return {WORD_ALIASES.get(t, t) for t in text.lower().split("_") if t}


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _normal_form(form: str) -> str:
    for old, new in FORM_ALIASES:
        form = form.replace(old, new)
    return form.strip("_")


def _match_score(form: str, species_ids: list[str]) -> int:
    """How well a form names a class: shared words, plus a bonus when the form's letters appear
    whole in a speciesId (POMPOM in oricorio_pom_pom)."""
    words = _tokens(form)
    best = 0
    for sid in species_ids:
        score = len(words & _tokens(sid)) + (2 if _alnum(form) and _alnum(form) in _alnum(sid) else 0)
        best = max(best, score)
    return best


def _priority(icon: Icon) -> tuple:
    """Plain art first, then cosmetic forms, shinies and costumes (by kind when assigned); Fandom's
    version of a look before the mined one."""
    rank = {"plain": 0, "form": 1, "shiny": 2, "costume": 3}.get(icon.kind)
    if rank is None:
        rank = 3 if icon.costume else 2 if icon.shiny else 1 if (icon.form or icon.gender) else 0
    return (rank, bool(icon.form), bool(icon.gender), icon.shiny, icon.source != "fandom", icon.name)


def _look(icon: Icon) -> tuple:
    """What an image shows, to keep one image per look across the two sources."""
    return (_alnum(_normal_form(icon.form) + icon.costume), bool(icon.gender), icon.shiny)


# How often each kind of image appears in a class's references, relative to the others:
#   plain    the class's own look (including its form: Galarian for stunfisk_galarian)
#   form     a cosmetic look the species has in GO (flower colors, a female look)
#   shiny    any look in shiny colors (but a costume)
#   costume  an event costume (hats, outfits): on teams least of all
KIND_WEIGHTS = {"plain": 6, "form": 3, "shiny": 2, "costume": 1}
_YEAR = re.compile(r"(19|20)\d\d")


def _kind(icon: Icon, matched: bool, cosmetic_forms: set[str]) -> str:
    form = _normal_form(icon.form)
    costume = bool(icon.costume) or bool(_YEAR.search(form))
    if form and not matched and not costume:
        # Fandom names costumes like forms: a costume is a look the mined icons don't list as a form
        costume = icon.source == "fandom" and _alnum(form) not in cosmetic_forms
    if costume:
        return "costume"
    if icon.shiny:
        return "shiny"
    if (form and not matched) or icon.gender:
        return "form"
    return "plain"


def assign(icons: list[Icon], classes: dict[int, list]) -> tuple[dict[int, list[Icon]], dict[str, int]]:
    """Images for each sprite class (`classes`: sid -> its Species), each with its kind, and
    counts of what was skipped or sent to a default class, for checking the mapping."""
    by_dex: dict[int, list[tuple[int, list[str]]]] = {}
    for sid, members in classes.items():
        by_dex.setdefault(members[0].dex, []).append((sid, [m.speciesId for m in members]))
    # The cosmetic forms each species has in GO, as the mined icons name them
    cosmetic: dict[int, set[str]] = {}
    for icon in icons:
        form = _normal_form(icon.form)
        if icon.source == "pogo" and form and not icon.costume and not _YEAR.search(form):
            cosmetic.setdefault(icon.dex, set()).add(_alnum(form))
    stats = {"matched": 0, "default": 0, "skipped": 0, "no class": 0}
    out: dict[int, list[Icon]] = {}
    for icon in icons:
        candidates = by_dex.get(icon.dex)
        if not candidates:
            stats["no class"] += 1
            continue
        form = _normal_form(icon.form)
        if _tokens(form) & SKIP_WORDS:
            stats["skipped"] += 1
            continue
        # A female look is a class of its own for some species (indeedee_female)
        if not form and icon.gender in ("female", "2"):
            form = "FEMALE"
        scored = [(_match_score(form, ids), -len(min(ids, key=len)), -sid, sid) for sid, ids in candidates]
        best = max(scored)
        matched = bool(form) and best[0] > 0
        if matched:
            stats["matched"] += 1
            target = best[3]
        else:
            # The default form, or a cosmetic one: the class with the plainest speciesId
            stats["default"] += 1
            target = min(candidates, key=lambda c: (len(_tokens(min(c[1], key=len))), c[0]))[0]
        # Matching the class but not its plainest speciesId (pikachu_flying in Pikachu's class) is
        # one of its variants, not its plain look
        ids = next(ids for sid, ids in candidates if sid == target)
        own = matched and _match_score(form, [min(ids, key=len)]) > 0
        kind = _kind(icon, own, cosmetic.get(icon.dex, set()))
        if matched and not own and kind == "plain":
            kind = "form"
        out.setdefault(target, []).append(replace(icon, kind=kind))
    chosen = {}
    for sid, found in out.items():
        seen, keep = set(), []
        for icon in sorted(found, key=_priority):
            if _look(icon) not in seen:
                seen.add(_look(icon))
                keep.append(icon)
        chosen[sid] = keep[:MAX_PER_CLASS]
    return chosen, stats


def list_icons() -> list[Icon]:
    """PokeMiners' icon names at the pinned commit, from GitHub's API (cached in INDEX)."""
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


def list_fandom(species: list) -> list[Icon]:
    """Fandom's sprites with their urls, from the wiki's API (cached in FANDOM_INDEX)."""
    if FANDOM_INDEX.exists():
        files = json.loads(FANDOM_INDEX.read_text())
    else:
        import requests

        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": FANDOM_CATEGORY,
            "gcmtype": "file",
            "gcmlimit": "500",
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
        files = []
        while True:
            response = requests.get(f"{FANDOM_API}?{urlencode(params)}", headers={"User-Agent": USER_AGENT}, timeout=60)
            response.raise_for_status()
            data = response.json()
            for page in data.get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("url"):
                    files.append({"title": page["title"].removeprefix("File:"), "url": info["url"]})
            if "continue" not in data:
                break
            params.update(data["continue"])
        FANDOM_INDEX.parent.mkdir(parents=True, exist_ok=True)
        FANDOM_INDEX.write_text(json.dumps(sorted(files, key=lambda f: f["title"])))
    names = species_dex(species)
    return [icon for icon in (parse_fandom(f["title"], f["url"], names) for f in files) if icon]


def download(classes: dict[int, list], workers: int = 8) -> dict[str, int]:
    """Download each class's images into GO_REFS/<sid>/, with the chosen ones listed in ORDER.
    Returns the mapping's counts."""
    import requests
    from tqdm import tqdm

    species = [m for members in classes.values() for m in members]
    assigned, stats = assign(list_fandom(species) + list_icons(), classes)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    jobs = [(icon, GO_REFS / str(sid) / icon.name) for sid, icons in assigned.items() for icon in icons]

    def fetch(job) -> bool:
        icon, dest = job
        if dest.exists() and dest.stat().st_size:
            return True
        url = icon.url or RAW_URL.format(name=quote(icon.name))
        for _ in range(3):
            try:
                response = session.get(url, timeout=30)
            except requests.RequestException:
                continue  # these hosts time out now and then
            if response.status_code != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(response.content)
            return True
        return False

    with ThreadPoolExecutor(workers) as pool:
        ok = list(tqdm(pool.map(fetch, jobs), total=len(jobs), desc="GO art", mininterval=5))
    fetched = {dest.name for (_, dest), good in zip(jobs, ok) if good}
    for sid, icons in assigned.items():
        folder = GO_REFS / str(sid)
        if folder.is_dir():
            (folder / ORDER).write_text(json.dumps([[i.name, i.kind] for i in icons if i.name in fetched]))
    fandom = sum(1 for icon, _ in jobs if icon.source == "fandom")
    return {
        **stats,
        "classes": len(assigned),
        "images": len(jobs),
        "fandom": fandom,
        "pogo": len(jobs) - fandom,
        "failed": ok.count(False),
    }


def _order(sid: int) -> list[tuple[str, str]]:
    """A class's chosen images and their kinds, best first."""
    folder = GO_REFS / str(sid)
    order = folder / ORDER
    if order.exists():
        entries = json.loads(order.read_text())
        # [name, kind]; plain names from before kinds were kept
        pairs = [(e, "") if isinstance(e, str) else (e[0], e[1]) for e in entries]
    else:  # downloaded before the order was kept: PokeMiners icons only
        icons = [i for i in (parse(p.name) for p in folder.glob("*.icon.png")) if i]
        pairs = [(i.name, "") for i in sorted(icons, key=_priority)]
    return [
        (name, kind or ("shiny" if name.endswith(".s.icon.png") or "_shiny" in name else "plain"))
        for name, kind in pairs
        if (folder / name).exists()
    ]


def reference_images(sid: int) -> list[np.ndarray]:
    """A class's GO images as RGBA arrays, best first, each repeated by its kind's weight
    (KIND_WEIGHTS): one pass over all of them, then passes over the heavier kinds."""
    folder = GO_REFS / str(sid)
    if not folder.is_dir():
        return []
    pairs = _order(sid)
    images = {name: np.asarray(Image.open(folder / name).convert("RGBA")) for name, _ in pairs}
    out = []
    for round_ in range(max(KIND_WEIGHTS.values())):
        out += [images[name] for name, kind in pairs if KIND_WEIGHTS[kind] > round_]
    return out


def plain_image(sid: int) -> np.ndarray | None:
    """A class's plain GO art, the look extra synthetic tiles are rendered from."""
    folder = GO_REFS / str(sid)
    if not folder.is_dir():
        return None
    pairs = _order(sid)
    name = next((n for n, kind in pairs if kind == "plain"), pairs[0][0] if pairs else None)
    if not name:
        return None
    rgba = np.asarray(Image.open(folder / name).convert("RGBA"))
    ys, xs = np.nonzero(rgba[..., 3] > 16)  # trimmed to the Pokémon, like the other references
    return rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] if len(ys) else None
