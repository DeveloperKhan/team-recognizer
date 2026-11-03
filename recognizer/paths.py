"""Project paths."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"


def _data_dir() -> Path:
    """Your own labeled data (screenshot labels, rosters, extra reference art): RECOGNIZER_DATA_DIR,
    else a sibling checkout ../team-recognizer-data, else local/ in this repo (gitignored)."""
    if os.environ.get("RECOGNIZER_DATA_DIR"):
        return Path(os.environ["RECOGNIZER_DATA_DIR"]).expanduser().resolve()
    sibling = ROOT.parent / "team-recognizer-data"
    return sibling if sibling.is_dir() else ROOT / "local"


DATA_DIR = _data_dir()
LABELS = DATA_DIR / "labels"  # screens.json, events.json
ROSTERS = DATA_DIR / "rosters"  # <event>.json
EXTRA_REFS = DATA_DIR / "refs" / "extra"  # <speciesId>/<image>
