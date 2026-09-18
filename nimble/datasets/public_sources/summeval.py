"""SummEval test split: expert consistency or relevance ratings of machine summaries, as five-level scores.

Each article row is unpacked into one raw dict per machine summary. The target is the three-expert
mean rounded half up; the votes are not published, so no distribution is emitted.
"""

import math

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "summeval"
SOURCE_URL = "https://huggingface.co/datasets/mteb/summeval"
LICENSE = "MIT"
NOTE = ("Test split unpacked to one record per machine summary. Target is the three-expert mean"
        " rounded half up to a level 1-5, stored as index 0-4; the raw mean is reference.expert_mean."
        " All summaries of one article share a family.")
RELEASE_YEAR = 2020
SUBSETS = ("consistency", "relevance")

DIMENSIONS = ("consistency", "relevance", "coherence", "fluency")
INSTRUCTIONS = {
    "consistency": ("How factually consistent is the summary with the article? Every statement in the"
                    " summary should be supported by the article."),
    "relevance": ("How well does the summary capture the important content of the article, without"
                  " unimportant or redundant material?"),
}
LEVELS = {
    "consistency": [
        "Multiple statements contradict or are absent from the article.",
        "At least one clear unsupported or contradicted statement.",
        "Mostly supported, with a minor unsupported detail.",
        "Supported, with at most a small imprecision.",
        "Every statement is supported by the article.",
    ],
    "relevance": [
        "Misses the main points or is mostly about minor details.",
        "Captures some key content but omits important points or includes much that is unimportant.",
        "Captures the main points with noticeable omissions or filler.",
        "Captures the main points with minor omissions.",
        "Captures all key content and only key content.",
    ],
}


def level_index(mean):
    """Round a 1-5 expert mean half up to a level and return its 0-based index."""
    if type(mean) not in (int, float) or not math.isfinite(mean) or not 1 <= mean <= 5:
        raise ValueError(f"Expert mean must be a finite number in [1, 5], got {mean!r}")
    return min(4, max(0, math.floor(mean + 0.5) - 1))


def rows(path, subset=""):
    """Unpack each article row into one raw dict per machine summary."""
    unpacked = []
    for raw in read_jsonl(path):
        if not isinstance(raw.get("id"), str) or not raw["id"].strip():
            raise ValueError("SummEval article rows need a nonempty id")
        if not isinstance(raw.get("text"), str) or not raw["text"].strip():
            raise ValueError(f"SummEval article {raw['id']} has no text")
        summaries = raw.get("machine_summaries")
        if not isinstance(summaries, list) or not summaries:
            raise ValueError(f"SummEval article {raw['id']} has no machine_summaries")
        for dimension in DIMENSIONS:
            values = raw.get(dimension)
            if not isinstance(values, list) or len(values) != len(summaries):
                raise ValueError(f"SummEval article {raw['id']}: {dimension} must have one value per summary")
        for index, summary in enumerate(summaries):
            unpacked.append({"article_id": raw["id"], "text": raw["text"], "summary_index": index,
                             "summary": summary,
                             **{dimension: raw[dimension][index] for dimension in DIMENSIONS}})
    return unpacked


def record(raw, subset=""):
    """Build one score record for the selected dimension of one machine summary."""
    for key in ("article_id", "text", "summary"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"SummEval row is missing a nonempty {key}")
    return record_for(
        f"summeval-{subset}-{raw['article_id']}-{raw['summary_index']}", "summeval-" + subset,
        raw["article_id"],
        {"article": raw["text"], "summary": raw["summary"]},
        LEVELS[subset], INSTRUCTIONS[subset], level_index(raw[subset]),
        {"source": "summeval", "dimension": subset, "expert_mean": raw[subset], "experts": 3,
         "summary_index": raw["summary_index"],
         "other_dimensions": {d: raw[d] for d in DIMENSIONS if d != subset}},
    )
