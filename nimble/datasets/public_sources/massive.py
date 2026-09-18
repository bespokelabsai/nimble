"""MASSIVE 1.1 test partition: route an assistant utterance to one of 18 scenarios (choice).

The MASSIVE id is shared across locales, so it is both the record id and the family and lets
de-DE be evaluated on exactly the en-US ids. Reads the upstream tarball or one locale's JSONL.
"""

import json
import tarfile
from pathlib import Path

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "massive"
SOURCE_URL = "https://huggingface.co/datasets/AmazonScience/massive"
LICENSE = "CC BY 4.0"
NOTE = ("Test partition of MASSIVE 1.1; scenario is the 18-way routing target. The MASSIVE id is the"
        " family, shared across locales for the same utterance.")
RELEASE_YEAR = 2022
SUBSETS = ("en-US", "de-DE")
PARTITION = "test"

INSTRUCTIONS = ("Which assistant domain should handle this utterance? Judge from the request itself, not"
                " from how it is phrased.")
CRITERIA = {
    "alarm": "Setting, changing, or querying alarms.",
    "audio": "Volume, mute, or audio output settings.",
    "calendar": "Events, meetings, reminders on a calendar.",
    "cooking": "Recipes and cooking instructions.",
    "datetime": "The current time, date, or time conversions.",
    "email": "Reading, sending, or checking email and contacts.",
    "general": "Small talk, jokes, greetings, or general assistant control such as repeat or confirm.",
    "iot": "Controlling lights, plugs, heating, cleaners, coffee machines, or wemo devices.",
    "lists": "Creating, querying, or removing list items.",
    "music": "Music preferences, likes, or music settings, not playing a specific item.",
    "news": "News headlines or updates.",
    "play": "Playing a specific song, podcast, radio, audiobook, or game.",
    "qa": "Factual questions, definitions, math, currency, stock prices.",
    "recommendation": "Suggestions for events, movies, or locations.",
    "social": "Social media posts or queries.",
    "takeaway": "Food orders and delivery status.",
    "transport": "Taxis, tickets, traffic, or travel directions.",
    "weather": "Weather forecasts or conditions.",
}


def rows(path, subset=""):
    """Read one locale's rows from the upstream tarball or from a single extracted JSONL file."""
    if subset not in SUBSETS:
        raise ValueError(f"MASSIVE needs --subset from {SUBSETS}, got {subset!r}")
    path = Path(path)
    if not path.name.endswith(".tar.gz"):
        return read_jsonl(path)
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith(f"/data/{subset}.jsonl")]
        if len(members) != 1:
            raise FileNotFoundError(f"Expected one data/{subset}.jsonl member in {path}")
        return [json.loads(line) for line in archive.extractfile(members[0]).read().decode().split("\n")
                if line.strip()]


def record(raw, subset=""):
    """Convert one MASSIVE row; rows outside the test partition or the requested locale are skipped."""
    if raw.get("partition") != PARTITION or raw.get("locale") != subset:
        return None
    scenario = raw["scenario"]
    if scenario not in CRITERIA:
        raise ValueError(f"MASSIVE row {raw.get('id')}: unknown scenario {scenario!r}")
    utterance = raw["utt"]
    if not isinstance(utterance, str) or not utterance.strip():
        return None
    return record_for(
        f"massive-{raw['id']}", f"massive-{subset}", f"massive-{raw['id']}",
        {"utterance": utterance, "locale": subset}, CRITERIA, INSTRUCTIONS, scenario,
        {"source": "massive", "locale": subset, "intent": raw.get("intent"), "massive_id": str(raw["id"])},
    )
