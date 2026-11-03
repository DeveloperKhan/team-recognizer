"""Build data/species.json from public sources: every released, non-shadow Pokémon GO species
(PvPoke's gamemaster ids, which Dracoviz uses too) with the sprite ids references are fetched by.

    uv run recognizer update-species

For each species:
  sid           Smogon's sprite index number (dex * 32 + forme); one sprite class per sid
  showdownSlug  the name Pokémon Showdown serves its sprites under, e.g. "ninetales-alola"

Sources: PvPoke's gamemaster, Showdown's pokedex.json, and Smogon's sprite index at a pinned
commit (removed upstream later; pinning also keeps sids stable). The speciesId -> Showdown
resolution mirrors the rules Dracoviz uses for its own gamemaster.
"""

from __future__ import annotations

import json
import re
import unicodedata

from .paths import DATA

PVPOKE_URL = "https://raw.githubusercontent.com/pvpoke/pvpoke/master/src/data/gamemaster/pokemon.json"
SMOGON_SPECIES_URL = (
    "https://raw.githubusercontent.com/smogon/sprites/6b9759b741e8d0375831328b64dbb2e31086ec91/data/species.json"
)
SHOWDOWN_POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"

# Full speciesId -> Showdown name, for the cases no rule below can reach.
SPECIES_ALIASES = {
    "minior_core": "minior",  # Showdown's plain "Minior" is the core form
    "cherrim_sunny": "cherrim_sunshine",
    "dudunsparce_two": "dudunsparce",
    "dudunsparce_three": "dudunsparce_three_segment",
    "tauros_combat": "tauros_paldea_combat",  # PvPoke leaves the region off Paldean Tauros
    "tauros_blaze": "tauros_paldea_blaze",
    "tauros_aqua": "tauros_paldea_aqua",
}

# Ordered rewrites applied to a speciesId before it is looked up.
SPECIES_REWRITES = [
    (re.compile(r"_alolan(?=_|$)"), "_alola"),
    (re.compile(r"_galarian(?=_|$)"), "_galar"),
    (re.compile(r"_hisuian(?=_|$)"), "_hisui"),
    (re.compile(r"_paldean(?=_|$)"), "_paldea"),
    (re.compile(r"_west_sea$"), ""),  # Showdown names the west sea form after the species
    (re.compile(r"_east_sea$"), "_east"),
    (re.compile(r"_family_of_"), "_"),
    (re.compile(r"_crowned_(sword|shield)$"), "_crowned"),
    (re.compile(r"_(ice|shadow)_rider$"), r"_\1"),
    (re.compile(r"_female$"), "_f"),
    (re.compile(r"_male$"), "_m"),
]

SEA_FORM_SPECIES = ("shellos", "gastrodon")

# Forms PvPoke lists once but that look different on the overlay, kept as their own species
# (and sprite classes) the way Dracoviz lists them: (speciesId, speciesName, PvPoke entry).
EXTRA_FORMS = [
    ("dudunsparce_two", "Dudunsparce_two", "dudunsparce"),
    ("dudunsparce_three", "Dudunsparce_three", "dudunsparce"),
    ("indeedee", "Indeedee", "indeedee_male"),
    ("squawkabilly_green", "Squawkabilly_green", "squawkabilly"),
    ("squawkabilly_blue", "Squawkabilly_blue", "squawkabilly"),
    ("squawkabilly_yellow", "Squawkabilly_yellow", "squawkabilly"),
    ("squawkabilly_white", "Squawkabilly_white", "squawkabilly"),
]

# Species with shadow forms in Pokémon GO that PvPoke's gamemaster doesn't mark (yet).
SHADOW_ELIGIBLE_EXTRA = {
    "grookey", "thwackey", "rillaboom", "scorbunny", "raboot", "cinderace",
    "sobble", "drizzile", "inteleon", "morelull",
}


def normalize_name(value: str) -> str:
    """Showdown's normalization: lowercase alphanumerics, accents dropped."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFD", value).lower())


def build_sid_index(species_json: dict) -> dict[str, int]:
    """Normalized Showdown name -> sid (the lowest sid where Smogon lists a name twice)."""
    index: dict[str, int] = {}
    for key, entry in species_json.items():
        sid = int(key[1:])
        full = f"{entry['base']}-{entry['forme']}" if entry.get("forme") else entry["base"]
        name = normalize_name(full)
        if name not in index or index[name] > sid:
            index[name] = sid
    return index


def build_base_forme_list(pokedex: dict) -> list[tuple[str, str]]:
    """(base species, default form) for species whose default form has a name, e.g.
    ("giratina", "altered"): PvPoke spells those forms out where Showdown doesn't."""
    formes = [
        (normalize_name(e["name"]), normalize_name(e["baseForme"]))
        for e in pokedex.values()
        if not e.get("baseSpecies") and e.get("baseForme")
    ]
    return sorted(formes, key=lambda f: -len(f[0]))


def make_resolver(sid_index: dict[str, int], base_formes: list[tuple[str, str]]):
    """speciesId -> (Showdown name, sid), or None."""

    def lookup(name: str):
        return (name, sid_index[name]) if name in sid_index else None

    def resolve(species_id: str):
        sid_ = re.sub(r"_shadow$", "", species_id)  # only a trailing suffix: calyrex_shadow_rider stays
        sid_ = SPECIES_ALIASES.get(sid_, sid_)
        for pattern, replacement in SPECIES_REWRITES:
            sid_ = pattern.sub(replacement, sid_)
        name = normalize_name(sid_)
        if (hit := lookup(name)) is not None:
            return hit
        for base, forme in base_formes:  # "giratina_altered" -> Giratina
            if len(name) > len(base) and name.startswith(base) and name.endswith(forme):
                if (hit := lookup(name[: len(name) - len(forme)])) is not None:
                    return hit
        parts = sid_.split("_")  # GO-only forms Showdown doesn't know: the plainest ancestor
        for i in range(len(parts) - 1, 0, -1):
            if (hit := lookup(normalize_name("_".join(parts[:i])))) is not None:
                return hit
        return None

    return resolve


def _slug(dex_entry: dict) -> str:
    """The file name Showdown serves a species' sprites under."""
    to_id = lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())  # noqa: E731
    if dex_entry.get("forme") and dex_entry.get("baseSpecies"):
        return f"{to_id(dex_entry['baseSpecies'])}-{to_id(dex_entry['forme'])}"
    return to_id(dex_entry["name"])


def build(pvpoke: list[dict], pokedex: dict, smogon: dict) -> list[dict]:
    resolve = make_resolver(build_sid_index(smogon), build_base_forme_list(pokedex))
    ids = {e["speciesId"] for e in pvpoke}
    out = []

    def add(species_id: str, name: str, entry: dict, lookup_id: str) -> None:
        hit = resolve(lookup_id)
        tags = entry.get("tags") or []
        out.append(
            {
                "speciesId": species_id,
                "speciesName": name,
                "dex": entry["dex"],
                "sid": hit[1] if hit else None,
                "showdownSlug": _slug(pokedex[hit[0]]) if hit and hit[0] in pokedex else None,
                "shadowEligible": "shadoweligible" in tags
                or f"{entry['speciesId']}_shadow" in ids
                or species_id in SHADOW_ELIGIBLE_EXTRA,
            }
        )

    for entry in pvpoke:
        species_id = entry["speciesId"]
        tags = entry.get("tags") or []
        if species_id.endswith("_shadow") or re.search(r"_(xs|xl)$", species_id) or "duplicate" in tags:
            continue
        if entry.get("released") is False:  # includes in-battle forms (formChange / originalFormId)
            continue
        if species_id in SEA_FORM_SPECIES:  # one PvPoke entry, two forms on screen
            for sea, label in (("east", "East"), ("west", "West")):
                sea_id = f"{species_id}_{sea}_sea"
                add(sea_id, f"{entry['speciesName']} ({label})", entry, sea_id)
            continue
        add(species_id, entry["speciesName"], entry, entry.get("aliasId") or species_id)
    by_id = {e["speciesId"]: e for e in pvpoke}
    for species_id, name, base in EXTRA_FORMS:
        if base in by_id and species_id not in {s["speciesId"] for s in out}:
            add(species_id, name, by_id[base], species_id)
    return sorted(out, key=lambda s: s["speciesId"])


def update() -> list[dict]:
    import requests

    fetch = lambda url: requests.get(url, timeout=60).json()  # noqa: E731
    species = build(fetch(PVPOKE_URL), fetch(SHOWDOWN_POKEDEX_URL), fetch(SMOGON_SPECIES_URL))
    (DATA / "species.json").write_text(json.dumps(species, indent=2, ensure_ascii=False) + "\n")
    return species
