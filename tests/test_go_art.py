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


ZYGARDE_AND_FRIENDS = {
    22976: species("zygarde", dex=718),
    22977: species("zygarde_10", dex=718),
    22980: species("zygarde_complete", dex=718),
    28032: species("indeedee", "indeedee_male", dex=876),
    28033: species("indeedee_female", dex=876),
    17760: species("darmanitan_standard", dex=555),
    17762: species("darmanitan_galarian_standard", dex=555),
}


def test_zygarde_indeedee_and_battle_forms():
    icons = [
        go_art.parse(n)
        for n in [
            "pm718.fTEN_PERCENT.icon.png",
            "pm718.fCOMPLETE_TEN_PERCENT.icon.png",
            "pm718.fFIFTY_PERCENT.icon.png",
            "pm876.fFEMALE.icon.png",
        ]
    ]
    names = go_art.species_dex(
        [SimpleNamespace(speciesName=n, dex=d) for n, d in [("Indeedee", 876), ("Darmanitan", 555), ("Zygarde", 718)]]
    )
    icons += [
        go_art.parse_fandom(t, "https://x/" + t, names)
        for t in [
            "Indeedee female.png",
            "Darmanitan zen galarian.png",
            "Darmanitan galarian.png",
            "Zygarde 10percent.png",
        ]
    ]
    assigned, _ = go_art.assign(icons, ZYGARDE_AND_FRIENDS)
    out = {sid: sorted(i.name for i in found) for sid, found in assigned.items()}
    assert out[22977] == ["fandom_Zygarde_10percent.png"]  # one image per look, Fandom's
    assert out[22980] == ["pm718.fCOMPLETE_TEN_PERCENT.icon.png"]
    assert out[22976] == ["pm718.fFIFTY_PERCENT.icon.png"]
    assert out[28033] == ["fandom_Indeedee_female.png", "pm876.fFEMALE.icon.png"]
    assert out[17762] == ["fandom_Darmanitan_galarian.png"]  # the Zen form is skipped
    assert 17760 not in out


def test_fandom_names():
    names = go_art.species_dex(
        [
            SimpleNamespace(speciesName=n, dex=d)
            for n, d in [("Nidoran Female", 29), ("Mr. Mime", 122), ("Flabébé", 669)]
        ]
    )
    nidoran = go_art.parse_fandom("Nidoran♀ shiny.png", "u", names)
    assert (nidoran.dex, nidoran.form, nidoran.shiny, nidoran.source) == (29, "", True, "fandom")
    assert go_art.parse_fandom("Mr. Mime galarian.png", "u", names).form == "GALARIAN"
    assert go_art.parse_fandom("Flabébé blue.png", "u", names).form == "BLUE"
    assert go_art.parse_fandom("Missingno.png", "u", names) is None
    assert go_art.parse_fandom("Mr. Mime cry.ogg", "u", names) is None


def test_fandom_is_preferred_for_the_same_look():
    names = go_art.species_dex([SimpleNamespace(speciesName="Florges", dex=671)])
    icons = [go_art.parse("pm671.fWHITE.icon.png"), go_art.parse_fandom("Florges white.png", "u", names)]
    out = assign_icons(icons)
    assert out[21472] == ["fandom_Florges_white.png"]


def assign_icons(icons):
    assigned, _ = go_art.assign(icons, CLASSES)
    return {sid: [i.name for i in found] for sid, found in assigned.items()}
