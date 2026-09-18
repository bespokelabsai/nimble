"""MultiNLI dev_matched from the original NYU zip, keeping the five annotator votes (choice).

The Hugging Face copy drops annotator_labels. Pairs whose gold_label is "-" had no majority and are skipped.
"""

from collections import Counter

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "multinli"
SOURCE_URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
LICENSE = "Mixed per genre: OANC public domain; fiction and other sources CC BY-SA 3.0 or similar"
NOTE = ("dev_matched pairs with five annotator votes each; reference.distribution is the vote share and"
        " the target is the released gold_label (majority). Pairs without a majority are skipped."
        " promptID groups the hypotheses written for one premise.")
RELEASE_YEAR = 2018
SUBSETS = ()
OPTIONS = ("entailment", "neutral", "contradiction")
CRITERIA = {
    "entailment": "If the premise is true, the hypothesis must be true.",
    "neutral": "The hypothesis might or might not be true; the premise does not settle it.",
    "contradiction": "If the premise is true, the hypothesis cannot be true.",
}
INSTRUCTIONS = "Assume the premise is true. How does the hypothesis relate to it?"


def rows(path, subset=""):
    return read_jsonl(path)


def record(raw, subset=""):
    """Build one record, or None for pairs without a majority label."""
    if raw.get("gold_label") == "-":
        return None
    for key in ("pairID", "promptID", "genre", "sentence1", "sentence2"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"MultiNLI row is missing a nonempty {key}")
    if raw["gold_label"] not in CRITERIA:
        raise ValueError(f"Unexpected MultiNLI gold_label: {raw['gold_label']!r}")
    votes = raw.get("annotator_labels")
    if not isinstance(votes, list) or not votes or any(v not in CRITERIA for v in votes):
        raise ValueError(f"{raw['pairID']}: annotator_labels must be nonempty MultiNLI labels")
    counts = Counter(votes)
    distribution = {option: counts[option] / len(votes) for option in OPTIONS}
    if distribution[raw["gold_label"]] != max(distribution.values()):
        raise ValueError(f"{raw['pairID']}: gold_label is not a most-voted option")
    return record_for(
        "multinli-" + raw["pairID"], "multinli-" + raw["genre"], raw["promptID"],
        {"premise": raw["sentence1"].strip(), "hypothesis": raw["sentence2"].strip()},
        CRITERIA, INSTRUCTIONS, raw["gold_label"],
        {"source": "multinli", "distribution": distribution, "annotations": len(votes),
         "unanimous": len(counts) == 1, "genre": raw["genre"]},
    )
