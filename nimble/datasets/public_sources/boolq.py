"""BoolQ validation split: yes/no questions over one Wikipedia passage (noul).

The passage text is the family key, since several questions can share a passage.
"""

import hashlib

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "boolq"
SOURCE_URL = "https://huggingface.co/datasets/google/boolq"
LICENSE = "CC BY-SA 3.0"
NOTE = "Yes/no questions over one Wikipedia passage; questions sharing a passage form one family."
RELEASE_YEAR = 2019
SUBSETS = ()

INSTRUCTIONS = ("Does the passage answer the question with yes? Use only what the passage states or"
                " directly implies.")
CRITERIA = {
    "true": "The passage states or directly implies that the answer is yes.",
    "false": "The passage states or directly implies that the answer is no.",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def rows(path, subset=""):
    return read_jsonl(path)


def record(raw, subset=""):
    """Build one BoolQ record, or None for rows without a boolean answer or the text fields."""
    if type(raw.get("answer")) is not bool:
        return None
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in ("question", "passage")):
        return None
    return record_for(
        "boolq-" + digest(raw["question"] + "\n" + raw["passage"]), "boolq", digest(raw["passage"]),
        {"passage": raw["passage"], "question": raw["question"]},
        CRITERIA, INSTRUCTIONS, raw["answer"], {"source": "boolq"},
    )
