"""Civil Comments test split: toxicity as the fraction of raters who called the comment toxic (noul).

The row position in the exported file is the identifier; the file has no id column.
"""

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "civil_comments"
SOURCE_URL = "https://huggingface.co/datasets/google/civil_comments"
LICENSE = "CC0 1.0"
THRESHOLD = 0.5
NOTE = (f"Test split; the target is toxicity >= {THRESHOLD}, the convention of the Jigsaw challenge."
        " reference.distribution is the rater fraction itself, so it is a point estimate of agreement"
        " rather than a vote count; the number of raters per comment is not published.")
RELEASE_YEAR = 2019
SUBSETS = ()
ATTRIBUTES = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack",
              "sexual_explicit")
CRITERIA = {
    "true": "The comment contains insults, threats, identity-based attacks, obscenity, or contempt aimed"
            " at a person or group.",
    "false": "The comment may disagree or criticise but does not attack, demean, or threaten.",
}
INSTRUCTIONS = ("Is this comment toxic, meaning rude, disrespectful, or unreasonable enough that a reader"
                " would likely leave the discussion?")


def rows(path, subset=""):
    return [{"row_index": index, **raw} for index, raw in enumerate(read_jsonl(path))]


def fraction(value, key):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"{key} must be a rater fraction in [0, 1], got {value!r}")
    return float(value)


def record(raw, subset=""):
    """Build one record; the row index within the exported file is the identifier."""
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    toxicity = fraction(raw.get("toxicity"), "toxicity")
    identifier = f"civil_comments-{raw['row_index']}"
    attributes = {key: fraction(raw[key], key) for key in ATTRIBUTES if key in raw}
    return record_for(
        identifier, "civil_comments", identifier, text.strip(),
        CRITERIA, INSTRUCTIONS, toxicity >= THRESHOLD,
        {"source": "civil_comments", "distribution": {"false": 1 - toxicity, "true": toxicity},
         "threshold": THRESHOLD, "rater_fractions": attributes},
    )
