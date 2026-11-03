"""The speciesId -> Showdown sprite resolution, and the committed species list it produces."""

import json

import numpy as np

from recognizer import species, train_head
from recognizer.paths import DATA

# Smogon's sprite index (s<sid> -> base/forme) and Showdown's pokedex, cut down to a few entries.
SMOGON = {
    "s1216": {"base": "Ninetales"},
    "s1217": {"base": "Ninetales", "forme": "Alola"},
    "s15584": {"base": "Giratina"},
    "s15585": {"base": "Giratina", "forme": "Origin"},
    "s28704": {"base": "Calyrex"},
    "s28706": {"base": "Calyrex", "forme": "Shadow"},
    "s13568": {"base": "Pikachu"},
}
POKEDEX = {
    "ninetales": {"name": "Ninetales"},
    "ninetalesalola": {"name": "Ninetales-Alola", "baseSpecies": "Ninetales", "forme": "Alola"},
    "giratina": {"name": "Giratina", "baseForme": "Altered"},
    "giratinaorigin": {"name": "Giratina-Origin", "baseSpecies": "Giratina", "forme": "Origin"},
    "calyrex": {"name": "Calyrex"},
    "calyrexshadow": {"name": "Calyrex-Shadow", "baseSpecies": "Calyrex", "forme": "Shadow"},
    "pikachu": {"name": "Pikachu"},
}


def resolve(species_id: str):
    return species.make_resolver(species.build_sid_index(SMOGON), species.build_base_forme_list(POKEDEX))(species_id)


def test_regional_forms_use_showdowns_names():
    assert resolve("ninetales_alolan") == ("ninetalesalola", 1217)


def test_shadow_suffix_is_dropped():
    assert resolve("ninetales_alolan_shadow") == ("ninetalesalola", 1217)


def test_named_default_form_maps_to_the_base_species():
    assert resolve("giratina_altered") == ("giratina", 15584)
    assert resolve("giratina_origin") == ("giratinaorigin", 15585)


def test_shadow_rider_is_a_form_not_a_shadow():
    assert resolve("calyrex_shadow_rider") == ("calyrexshadow", 28706)


def test_go_only_forms_fall_back_to_the_plainest_ancestor():
    assert resolve("pikachu_libre") == ("pikachu", 13568)


def test_unknown_species_resolve_to_none():
    assert resolve("missingno") is None


def test_committed_species_list():
    entries = json.loads((DATA / "species.json").read_text())
    ids = [e["speciesId"] for e in entries]
    assert len(ids) == len(set(ids))
    assert not [i for i in ids if i.endswith("_shadow")]
    assert all(isinstance(e["sid"], int) and e["showdownSlug"] for e in entries)


def test_species_head_classes_are_in_the_species_list():
    """After `update-species`, every class the trained head scores must still have a species."""
    sids = {e["sid"] for e in json.loads((DATA / "species.json").read_text())}
    _, classes = train_head.load()
    missing = sorted(set(np.asarray(classes).tolist()) - sids)
    assert not missing, f"head classes without a species: {missing[:10]}"
