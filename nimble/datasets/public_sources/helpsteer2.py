"""HelpSteer2 validation split: human helpfulness 0-4 of an assistant response, as a five-level score.

Both responses to a prompt share a family. The other four attributes stay under reference only.
"""

import hashlib

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "helpsteer2"
SOURCE_URL = "https://huggingface.co/datasets/nvidia/HelpSteer2"
LICENSE = "CC BY-4.0"
NOTE = ("Validation split; helpfulness 0-4 scored as level index 0-4. Two responses per prompt"
        " share a family. correctness, coherence, complexity, and verbosity are kept in reference only.")
RELEASE_YEAR = 2024
SUBSETS = ()

ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
LEVELS = [
    "Not helpful: ignores or misreads the prompt, or is wrong in ways that make it useless.",
    "Slightly helpful: touches the request but is mostly incorrect, incomplete, or off target.",
    "Partially helpful: addresses the request with notable gaps or errors a user would need to fix.",
    "Mostly helpful: addresses the request well with minor omissions or imperfections.",
    "Extremely helpful: fully and accurately addresses the request; nothing important is missing.",
]
INSTRUCTIONS = ("How helpful is the response to the prompt, considering whether it addresses what was"
                " asked, is correct, and is complete?")


def family_key(prompt):
    return "helpsteer2-" + hashlib.sha256(prompt.encode()).hexdigest()[:16]


def rows(path, subset=""):
    """Read the exported validation JSONL; the file position becomes each row's stable index."""
    return [{**raw, "row_index": index} for index, raw in enumerate(read_jsonl(path))]


def record(raw, subset=""):
    """Build one score record; both responses to a prompt share the prompt-hash family."""
    for key in ("prompt", "response"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"HelpSteer2 row is missing a nonempty {key}")
    for key in ATTRIBUTES:
        if type(raw.get(key)) is not int or not 0 <= raw[key] <= 4:
            raise ValueError(f"HelpSteer2 {key} must be an integer 0-4, got {raw.get(key)!r}")
    return record_for(
        f"helpsteer2-{raw['row_index']}", "helpsteer2", family_key(raw["prompt"]),
        {"prompt": raw["prompt"], "response": raw["response"]},
        LEVELS, INSTRUCTIONS, raw["helpfulness"],
        {"source": "helpsteer2", "row_index": raw["row_index"],
         **{key: raw[key] for key in ATTRIBUTES if key != "helpfulness"}},
    )
