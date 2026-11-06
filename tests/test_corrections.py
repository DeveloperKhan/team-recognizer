"""Shared corrections: which tiles of a submission train, review decisions, and dedupe."""

import json

import cv2
import numpy as np
import pytest

from recognizer import corrections


def tile_box(i: int) -> list[float]:
    col, row = i % 2, i // 2
    return [0.05 + col * 0.45, 0.2 + row * 0.26, 0.45 + col * 0.45, 0.44 + row * 0.26]


def slot(species: str, confidence: float, edited: bool = False, shadow: bool = False, i: int = 0) -> dict:
    predicted = {"speciesId": "kingdra" if edited else species, "shadow": False, "confidence": confidence}
    return {"speciesId": species, "shadow": shadow, "edited": edited, "predicted": predicted, "box": tile_box(i)}


def write_submission(root, name: str, url: str = "https://x.challonge.com/2027_GO_Baltimore", seed: int = 0):
    folder = root / "inbox" / name
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(folder / "team-0.jpg"), rng.integers(0, 255, (400, 300, 3), dtype=np.uint8))
    slots = [
        slot("corsola", 1.0, edited=True, shadow=True, i=0),  # changed by hand
        slot("melmetal", 0.98, i=1),  # confident
        slot("stunfisk", 0.51, i=2),  # unchecked
        slot("", 0.0, i=3),  # left unrecognized
        slot("azumarill", 0.95, i=4),
        slot("hydreigon", 0.9, shadow=True, i=5),
    ]
    team = {
        "image": "team-0.jpg",
        "side": "right",
        "read": "SCEPTILEICE25",
        "dracovizName": None,
        "player": {"name": "Sceptileice25", "challongeName": "Sceptileice25", "rank": 2},
        "nameMatch": {"via": "read", "edits": 0},
        "nameBox": [0.05, 0.02, 0.95, 0.15],
        "source": "shot.png (right)",
        "engine": "browser",
        "modelVersion": "01e6f3e",
        "slots": slots,
    }
    labels = {"format": 1, "createdAt": "2026-11-05T00:00:00Z", "event": {"challongeUrl": url, "name": "Baltimore"}}
    (folder / "labels.json").write_text(json.dumps({**labels, "via": "upload", "teams": [team]}))
    return folder


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(corrections, "CORRECTIONS", tmp_path)
    return tmp_path


def test_event_slug_matches_event_names():
    assert (
        corrections.event_slug("https://pokemongochampionshipseries.challonge.com/2027_GO_Baltimore")
        == "2027-baltimore"
    )
    assert (
        corrections.event_slug("https://pokemongochampionshipseries.challonge.com/2027_GO_Gdansk/standings")
        == "2027-gdansk"
    )
    assert corrections.event_slug("https://challonge.com/abc123") == "abc123"
    assert corrections.event_slug(None) == "unknown"


def test_which_tiles_train(root):
    folder = write_submission(root, "a")
    tiles = corrections.tiles(folder)
    assert [(t.species_id, t.kind, t.weight) for t in tiles] == [
        ("corsola", "edited", 2.0),
        ("melmetal", "confident", 1.0),
        ("azumarill", "confident", 1.0),
        ("hydreigon", "confident", 1.0),
    ]
    assert tiles[0].shadow and tiles[0].event == "2027-baltimore"
    assert all(t.tile.shape[0] > 50 and t.tile.shape[1] > 50 for t in tiles)
    # Accepting with --unchecked also trusts the low-confidence tile left as recognized
    assert "stunfisk" in [t.species_id for t in corrections.tiles(folder, unchecked=True)]


def test_only_accepted_submissions_train_and_duplicates_count_once(root):
    write_submission(root, "a", seed=1)
    write_submission(root, "b", seed=1)  # the same panel shared twice
    write_submission(root, "c", seed=2)
    assert corrections.training_tiles() == []
    for name in ("a", "b"):
        corrections.move(name, "accepted")
    corrections.move("c", "rejected")
    assert len(corrections.training_tiles()) == 4
    assert corrections.training_tiles(exclude_events=("2027-baltimore",)) == []

    corrections.move("a", "accepted", unchecked=True)
    labels = json.loads((root / "accepted" / "a" / "labels.json").read_text())
    assert labels["review"]["state"] == "accepted" and labels["review"]["unchecked"] is True
    assert len(corrections.training_tiles()) == 5
    counts = corrections.summary()
    assert counts["accepted"]["submissions"] == 2 and counts["rejected"]["submissions"] == 1


def test_review_sheet(root, tmp_path):
    write_submission(root, "2026-11-05-baltimore-abc123")
    sheet = corrections.review("inbox", out=tmp_path / "review.html").read_text()
    assert "2026-11-05-baltimore-abc123" in sheet
    assert sheet.count("data:image/jpeg;base64,") == 7  # the name bar and six tiles
    assert "was kingdra" in sheet and "unchecked" in sheet


def test_rejects_unknown_formats(root):
    folder = write_submission(root, "a")
    labels = json.loads((folder / "labels.json").read_text())
    (folder / "labels.json").write_text(json.dumps({**labels, "format": 99}))
    with pytest.raises(ValueError):
        corrections.tiles(folder)


def write_submission_v2(root, name: str, seed: int = 0):
    """Format 2: only the tiles worth learning from, each its own crop, plus the name bar."""
    folder = root / "inbox" / name
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(folder / "team-0-name.jpg"), rng.integers(0, 255, (40, 200, 3), dtype=np.uint8))
    tiles = [
        (4, "forretress", True, True, ["edited", "species", "shadow"]),  # fixed by hand
        (5, "carbink", False, False, ["species"]),  # unsure, left as recognized
        (1, "toxapex", False, False, ["shadow"]),
    ]
    slots = []
    for index, species, shadow, edited, reasons in tiles:
        image = f"team-0-slot-{index}.jpg"
        cv2.imwrite(str(folder / image), rng.integers(0, 255, (120, 110, 3), dtype=np.uint8))
        predicted = {
            "speciesId": species,
            "shadow": False,
            "confidence": 0.6,
            "shadowViolet": 0.004,
            "shadowProbe": 0.55,
        }
        slots.append(
            {
                "index": index,
                "image": image,
                "box": [0.083, 0.083, 0.917, 0.917],
                "speciesId": species,
                "shadow": shadow,
                "edited": edited,
                "reasons": reasons,
                "predicted": predicted,
            }
        )
    team = {
        "nameImage": "team-0-name.jpg",
        "side": "left",
        "read": "OUTOFPOKET",
        "dracovizName": None,
        "player": {"name": "OutOfPoket", "challongeName": "HotPoket777", "rank": 11},
        "nameMatch": {"via": "read", "edits": 0},
        "source": "frame.png (left)",
        "engine": "browser",
        "modelVersion": "01e6f3e",
        "slots": slots,
    }
    labels = {
        "format": 2,
        "createdAt": "2026-11-06T00:00:00Z",
        "event": {"challongeUrl": "https://x.challonge.com/2027_GO_Gdansk", "name": "Gdańsk"},
        "via": "discord",
        "teams": [team],
    }
    (folder / "labels.json").write_text(json.dumps(labels))
    return folder


def test_format_2_trains_edits_and_only_trusted_unsure_tiles(root):
    folder = write_submission_v2(root, "v2")
    assert [(t.species_id, t.slot, t.kind, t.weight) for t in corrections.tiles(folder)] == [
        ("forretress", 4, "edited", 2.0)
    ]
    trusted = corrections.tiles(folder, unchecked=True, skip=["0:1"])
    assert [(t.species_id, t.slot) for t in trusted] == [("forretress", 4), ("carbink", 5)]
    assert trusted[0].shadow and trusted[0].event == "2027-gdansk"

    corrections.move("v2", "accepted", unchecked=True, skip=["0:1"])
    labels = json.loads((root / "accepted" / "v2" / "labels.json").read_text())
    assert labels["review"]["skip"] == ["0:1"]
    assert [t.species_id for t in corrections.training_tiles()] == ["forretress", "carbink"]
    assert corrections.summary()["accepted"] == {
        "submissions": 1,
        "teams": 1,
        "edited": 1,
        "confident": 0,
        "unchecked": 2,
    }


def test_format_2_review_sheet_and_score(root, tmp_path):
    write_submission_v2(root, "v2")
    sheet = corrections.review("inbox", out=tmp_path / "review.html").read_text()
    assert sheet.count("data:image/jpeg;base64,") == 4  # the name bar and three tiles
    assert "0:4 · edited, species, shadow" in sheet and "0:5 · species" in sheet

    class Recognizer:  # reads every tile as a non-shadow forretress
        def classify_tiles(self, tiles):
            return [{"speciesId": "forretress", "shadow": False, "confidence": 0.9} for _ in tiles]

    corrections.move("v2", "accepted")
    stats = corrections.score(Recognizer())["2027-gdansk"]
    assert (stats["tiles"], stats["species"], stats["shadow"]) == (3, 1, 2)
    assert (stats["edited"], stats["edited ok"]) == (1, 0)
