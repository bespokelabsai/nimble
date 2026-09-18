"""SQuAD 2.0 dev: does the paragraph state the answer to the question (noul).

Accepts the flat Hugging Face JSONL export or the nested official dev-v2.0.json. A paragraph's
questions form one family; answer text and offsets stay out of the model input.
"""

import hashlib
import json
from pathlib import Path

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "squad2"
SOURCE_URL = "https://huggingface.co/datasets/rajpurkar/squad_v2"
LICENSE = "CC BY-SA 4.0"
NOTE = ("Answerability of a question from one Wikipedia paragraph; the questions on a paragraph form one"
        " family, and answer text never enters the model input.")
RELEASE_YEAR = 2018
SUBSETS = ()

INSTRUCTIONS = ("Does the paragraph contain the information needed to answer the question? Treat a"
                " question as unanswerable when the paragraph discusses the topic but does not state the"
                " specific fact asked for.")
CRITERIA = {
    "true": "The paragraph states the answer explicitly.",
    "false": ("The paragraph does not state the answer, even if it covers the same subject or contains a"
              " plausible-looking but incorrect match."),
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def flatten(document):
    """One dict per question from the nested official layout, in file order."""
    for article in document["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                yield {"id": qa["id"], "title": article.get("title", ""), "context": paragraph["context"],
                       "question": qa["question"], "answers": qa.get("answers", []),
                       "is_impossible": qa.get("is_impossible")}


def rows(path, subset=""):
    path = Path(path)
    if path.suffix == ".json":
        return list(flatten(json.loads(path.read_text(encoding="utf-8"))))
    return read_jsonl(path)


def answer_texts(raw):
    """Distinct answer strings from either layout; empty means unanswerable."""
    answers = raw.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text", [])
    elif isinstance(answers, list):
        texts = [answer.get("text") for answer in answers if isinstance(answer, dict)]
    else:
        return None
    if any(not isinstance(text, str) for text in texts):
        return None
    return sorted(set(text for text in texts if text.strip()))


def record(raw, subset=""):
    """Build one answerability record; None when the row lacks the text fields or a usable label."""
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in ("id", "context", "question")):
        return None
    texts = answer_texts(raw)
    if texts is None:
        return None
    answerable = bool(texts)
    if raw.get("is_impossible") is not None and bool(raw["is_impossible"]) == answerable:
        return None  # The official flag and the answer list disagree; do not guess.
    return record_for(
        "squad2-" + raw["id"], "squad2", digest(raw["context"]),
        {"paragraph": raw["context"], "question": raw["question"]},
        CRITERIA, INSTRUCTIONS, answerable,
        {"source": "squad2", "title": raw.get("title", ""), "answer_texts": texts},
    )
