"""PubMedQA pqa_labeled: answer a biomedical question from its abstract with yes, no, or maybe (choice).

The long answer states the decision outright, so it never enters the record; section labels and
MeSH terms are omitted as well.
"""

from nimble.datasets.public_records import read_jsonl, record_for

NAME = "pubmedqa"
SOURCE_URL = "https://huggingface.co/datasets/qiaojin/PubMedQA"
LICENSE = "MIT"
NOTE = ("The 1,000-instance expert-labeled subset (pqa_labeled). State is the question and the joined"
        " abstract contexts; the long answer is excluded because it states the decision.")
RELEASE_YEAR = 2019
SUBSETS = ()

INSTRUCTIONS = "Based only on the abstract, what is the answer to the research question?"
CRITERIA = {
    "yes": "The abstract's findings support a yes answer.",
    "no": "The abstract's findings support a no answer.",
    "maybe": "The abstract's findings are mixed, conditional, or insufficient to answer either way.",
}


def rows(path, subset=""):
    return read_jsonl(path)


def contexts_of(raw):
    context = raw.get("context")
    if isinstance(context, dict):
        context = context.get("contexts", [])
    if isinstance(context, str):
        context = [context]
    return [c for c in context or [] if isinstance(c, str) and c.strip()]


def record(raw, subset=""):
    """Convert one labeled row; rows without a decision or an abstract are skipped."""
    decision = raw.get("final_decision")
    decision = decision.strip().lower() if isinstance(decision, str) else None
    contexts = contexts_of(raw)
    question = raw.get("question")
    if decision not in CRITERIA or not contexts or not isinstance(question, str) or not question.strip():
        return None
    pubid = str(raw["pubid"])
    return record_for(
        f"pubmedqa-{pubid}", "pubmedqa", pubid,
        {"question": question, "abstract_context": " ".join(contexts)}, CRITERIA, INSTRUCTIONS, decision,
        {"source": "pubmedqa", "pubid": pubid, "context_sections": len(contexts)},
    )
