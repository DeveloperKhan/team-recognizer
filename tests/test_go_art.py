"""Which sprite class each of Pokémon GO's icons goes to."""

from types import SimpleNamespace

from recognizer import go_art


def species(*ids: str, dex: int):
    return [SimpleNamespace(speciesId=i, dex=dex) for i in ids]


# sid -> the class's Species, the way refs.sprite_classes() gives them
CLASSES = {
    19776: species("stunfisk", dex=618),
    19777: species("stunfisk_galarian", dex=618),
    4096: species("tauros", dex=128),
    4097: species("tauros_combat", dex=128),
    4098: species("tauros_blaze", dex=128),
    15584: species("giratina_altered", dex=487),
    15585: species("giratina_origin", dex=487),
    23712: species("oricorio_baile", dex=741),
    23713: species("oricorio_pom_pom", dex=741),
    21472: species("florges", dex=671),
}


def assign(*names: str) -> dict[int, list[str]]:
    assigned, _ = go_art.assign([go_art.parse(n) for n in names], CLASSES)
    return {sid: [i.name for i in icons] for sid, icons in assigned.items()}


def test_parse():
    icon = go_art.parse("pm618.fGALARIAN.cFALL_2019.g2.s.icon.png")
    assert (icon.dex, icon.form, icon.costume, icon.gender, icon.shiny) == (618, "GALARIAN", "FALL_2019", "2", True)
    assert go_art.parse("pm618.icon.png").form == ""
    assert go_art.parse("not-an-icon.png") is None


def test_forms_go_to_their_classes():
    out = assign(
        "pm618.icon.png",
        "pm618.fGALARIAN.icon.png",
        "pm128.fPALDEA_COMBAT.icon.png",
        "pm128.fPALDEA_BLAZE.s.icon.png",
        "pm487.fORIGIN.icon.png",
        "pm487.fALTERED.icon.png",
        "pm741.fPOMPOM.icon.png",
    )
    assert out[19776] == ["pm618.icon.png"]
    assert out[19777] == ["pm618.fGALARIAN.icon.png"]
    assert out[4097] == ["pm128.fPALDEA_COMBAT.icon.png"]
    assert out[4098] == ["pm128.fPALDEA_BLAZE.s.icon.png"]
    assert out[15585] == ["pm487.fORIGIN.icon.png"]
    assert out[15584] == ["pm487.fALTERED.icon.png"]
    assert out[23713] == ["pm741.fPOMPOM.icon.png"]


def test_cosmetic_looks_go_to_the_default_class_and_others_are_skipped():
    out = assign(
        "pm671.fWHITE.icon.png",
        "pm671.fRED.s.icon.png",
        "pm128.cFALL_2019.icon.png",
        "pm618.fMEGA.icon.png",  # never on a team
        "pm9999.icon.png",  # no class
    )
    assert sorted(out[21472]) == ["pm671.fRED.s.icon.png", "pm671.fWHITE.icon.png"]
    assert out[4096] == ["pm128.cFALL_2019.icon.png"]
    assert 19776 not in out and 19777 not in out


def test_plain_art_comes_first_and_is_capped():
    names = [f"pm671.cEVENT_{i}.icon.png" for i in range(20)] + ["pm671.s.icon.png", "pm671.icon.png"]
    out = assign(*names)[21472]
    assert out[:2] == ["pm671.icon.png", "pm671.s.icon.png"]
    assert len(out) == go_art.MAX_PER_CLASS
