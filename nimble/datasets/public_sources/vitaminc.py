"""VitaminC dev: contrastive fact verification from Wikipedia revisions (choice).

case_id groups the siblings of one revision and is the family key.
"""

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "vitaminc"
SOURCE_URL = "https://huggingface.co/datasets/tals/vitaminc"
LICENSE = "CC BY-SA 3.0"
NOTE = "Contrastive Wikipedia revisions; case_id groups the siblings of one revision."
RELEASE_YEAR = 2021
SUBSETS = ()

CRITERIA = {
    "SUPPORTS": "The evidence states the claim, or the claim follows directly from the evidence.",
    "REFUTES": "The evidence states the opposite of the claim, or the claim is directly contradicted by it.",
    "NOT ENOUGH INFO": "The evidence neither establishes nor contradicts the claim. Settling it would need a"
                       " fact the evidence does not supply.",
}
INSTRUCTIONS = ("Decide how the evidence bears on the claim. Judge only from the evidence text, and not"
                " from outside knowledge about the subject.")


def rows(path, subset=""):
    return read_jsonl(path)


def record(raw, subset=""):
    """Build one VitaminC record; case_id keeps contrastive siblings in one family."""
    for key in ("unique_id", "case_id", "label", "claim", "evidence", "revision_type"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"VitaminC row is missing a nonempty {key}")
    if raw["label"] not in CRITERIA:
        raise ValueError(f"Unexpected VitaminC label: {raw['label']!r}")
    return record_for(
        "vitaminc-" + raw["unique_id"], "vitaminc-" + raw["revision_type"], raw["case_id"],
        {"evidence": raw["evidence"], "claim": raw["claim"]},
        CRITERIA, INSTRUCTIONS, raw["label"],
        {"source": "vitaminc", "page": raw.get("page", ""), "revision_type": raw["revision_type"],
         "wiki_revision_id": raw.get("wiki_revision_id", "")},
    )
