"""Record assembly and leak checks shared by every public-benchmark source module.

Source modules under nimble.datasets.public_sources import this module, never
public_benchmarks, so discovery cannot form an import cycle.
"""

import hashlib
import json
from pathlib import Path

from nimble.evaluation.evaluate_pilot import adapt_input

# Any of these keys reachable from a model input means reference material leaked.
FORBIDDEN_INPUT_KEYS = {"target", "distribution", "reference", "label", "labels", "majority_label",
                        "annotations", "label_dist", "label_count", "label_counter", "old_label",
                        "old_labels", "entropy", "gold", "answers", "rating", "score", "long_answer",
                        "final_decision"}


def reference_keys(value):
    """Collect every mapping key reachable in a model input, at any depth."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in reference_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in reference_keys(item)}
    return set()


def question_kind(criteria, target):
    """Infer choice, noul, or score from the criteria shape."""
    if isinstance(criteria, list):
        return "score"
    if isinstance(criteria, dict) and set(criteria) == {"false", "true"} and type(target) is bool:
        return "noul"
    if isinstance(criteria, dict):
        return "choice"
    raise ValueError("criteria must be a mapping of options or an ordered list of levels")


def candidate_keys(kind, criteria):
    """The keys a scorer's probability dictionary uses for this question."""
    if kind == "score":
        return [str(i) for i in range(len(criteria))]
    return list(criteria)


def record_for(identifier, domain, family, state, criteria, instructions, target, reference):
    """Assemble one record and reject any reference material that leaked into the model input.

    Choice: criteria maps option -> description, target is an option key.
    Noul: criteria is keyed exactly "false" and "true", target is a bool.
    Score: criteria is an ordered list of level descriptions, target is the level index.
    reference may carry "distribution" (keys as the scorer reports them) and "annotations".
    """
    kind = question_kind(criteria, target)
    if kind == "choice":
        if target not in criteria:
            raise ValueError(f"{identifier}: target {target!r} is not one of the offered options")
        question = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
    elif kind == "noul":
        if any(not isinstance(text, str) or not text.strip() for text in criteria.values()):
            raise ValueError(f"{identifier}: noul criteria need nonempty false and true descriptions")
        question = {"type": "noul", "instructions": instructions,
                    "criteria": {"false": criteria["false"], "true": criteria["true"]}}
    else:
        if type(target) is not int or not 0 <= target < len(criteria):
            raise ValueError(f"{identifier}: score target must be a level index within the criteria")
        if len(criteria) < 2 or any(not isinstance(text, str) or not text.strip() for text in criteria):
            raise ValueError(f"{identifier}: score criteria need at least two nonempty ordered levels")
        question = {"type": "score", "instructions": instructions, "criteria": list(criteria)}
    model_input = {"state": state, "questions": {"decision": question}}
    # The target is legitimately one of the offered option keys, so check the structure
    # instead: no reference-bearing key may be reachable from the input.
    leaked = reference_keys(model_input) & FORBIDDEN_INPUT_KEYS
    if leaked:
        raise ValueError(f"{identifier}: reference keys reached the model input: {sorted(leaked)}")
    if "distribution" in reference:
        distribution = reference["distribution"]
        keys = candidate_keys(kind, criteria)
        if not isinstance(distribution, dict) or set(distribution) != set(keys):
            raise ValueError(f"{identifier}: distribution keys must be exactly {keys}")
        values = list(distribution.values())
        if any(type(v) not in (int, float) or v < 0 for v in values) or abs(sum(values) - 1) > 1e-6:
            raise ValueError(f"{identifier}: distribution must be nonnegative and sum to one")
    row = {
        "id": identifier, "split": "validation", "domain": domain, "family": family, "source_family": family,
        "input": model_input,
        "reference": {"target": target, "human_reviewed": True, **reference},
    }
    adapt_input(row["input"])
    return row


def read_jsonl(path):
    """Read one JSON object per newline-delimited line; blank lines are ignored.

    Split on "\\n" only: str.splitlines() also breaks on U+2028/U+2029 and other
    Unicode separators, which occur inside real text fields (a PubMedQA abstract
    contains U+2029) and would corrupt the parse.
    """
    with open(path, encoding="utf-8", newline="\n") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def source_digest(path):
    """SHA-256 of the downloaded upstream file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
