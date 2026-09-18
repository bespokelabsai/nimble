"""PAWS labeled_final test split: adversarial paraphrase pairs with high word overlap (noul).
"""

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "paws"
SOURCE_URL = "https://huggingface.co/datasets/google-research-datasets/paws"
LICENSE = "Google PAWS license: may be freely used for any purpose"
NOTE = "Adversarial paraphrase pairs with high word overlap; each labeled_final test pair is one family."
RELEASE_YEAR = 2019
SUBSETS = ()

INSTRUCTIONS = ("Do the two sentences have the same meaning? Sentences that reuse the same words in a"
                " different order can still mean different things.")
CRITERIA = {
    "true": "Both sentences describe the same situation with the same participants in the same roles.",
    "false": ("The sentences differ in meaning, including cases where the same words are rearranged so"
              " that who does what to whom changes."),
}


def rows(path, subset=""):
    return read_jsonl(path)


def record(raw, subset=""):
    """Build one PAWS record, or None when the label is not 0/1 or a sentence is missing."""
    if raw.get("label") not in (0, 1) or type(raw.get("label")) is bool:
        return None
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in ("sentence1", "sentence2")):
        return None
    if type(raw.get("id")) not in (int, str):
        return None
    identifier = str(raw["id"])
    return record_for(
        "paws-" + identifier, "paws", identifier,
        {"sentence_1": raw["sentence1"], "sentence_2": raw["sentence2"]},
        CRITERIA, INSTRUCTIONS, raw["label"] == 1, {"source": "paws"},
    )
