"""Compare saved public-benchmark runs on identical records, including contrastive family accuracy."""

import argparse
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from nimble.datasets.public_records import read_jsonl
from nimble.evaluation.evaluate_public import expected_calibration_error

Z95 = 1.96


def wilson_interval(correct, total, z=Z95):
    """Wilson score interval for a binomial proportion, as in evaluate_banking77."""
    if total < 1 or not 0 <= correct <= total:
        raise ValueError("Need 0 <= correct <= total with a positive total")
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def mcnemar_exact(b, c):
    """Exact two-sided binomial test on the discordant counts under the null of equal error rates."""
    if b < 0 or c < 0:
        raise ValueError("Discordant counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    low = min(b, c)
    tail = sum(math.comb(n, k) for k in range(low + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_rows(path):
    rows = read_jsonl(path)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: rows must have unique IDs")
    return {r["id"]: r for r in rows}


def join_runs(runs):
    """Require every run to cover the same IDs with the same references and question types."""
    if len(runs) < 2:
        raise ValueError("Comparison needs at least two runs")
    names = list(runs)
    ids = set(runs[names[0]])
    for name in names[1:]:
        if set(runs[name]) != ids:
            raise ValueError(f"{names[0]} and {name} cover different record IDs")
    for row_id in ids:
        first = runs[names[0]][row_id]
        for name in names[1:]:
            other = runs[name][row_id]
            if other["reference"]["target"] != first["reference"]["target"] or other["type"] != first["type"]:
                raise ValueError(f"{row_id}: reference or type differs between {names[0]} and {name}")
    return sorted(ids)


def family_complete(rows):
    """Fraction of families in which every record was answered correctly."""
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row["student"]["correct"])
    complete = sum(all(flags) for flags in families.values())
    return {"families": len(families), "complete": complete,
            "accuracy": complete / len(families) if families else None,
            "sizes": {str(k): v for k, v in sorted(Counter(len(f) for f in families.values()).items())}}


def accuracy_block(rows):
    correct = sum(bool(r["student"]["correct"]) for r in rows)
    return {"count": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "accuracy_ci95": wilson_interval(correct, len(rows))}


def run_metrics(rows):
    """Per-run accuracy, calibration, and probabilistic quality; invalid responses count as wrong."""
    valid = [r for r in rows if "error" not in r]
    result = {**accuracy_block(rows), "valid": len(valid), "errors": len(rows) - len(valid),
              "family_complete": family_complete(rows),
              "by_type": {kind: accuracy_block([r for r in rows if r["type"] == kind])
                          for kind in sorted({r["type"] for r in rows})},
              "by_domain": {domain: accuracy_block([r for r in rows if r["domain"] == domain])
                            for domain in sorted({r["domain"] for r in rows})}}
    if valid:
        result["ece10"] = expected_calibration_error(
            (r["student"]["top_probability"], r["student"]["correct"]) for r in valid)
        result["mean_negative_log_likelihood"] = statistics.mean(
            r["student"]["negative_log_likelihood"] for r in valid)
        result["mean_multiclass_brier"] = statistics.mean(r["student"]["multiclass_brier"] for r in valid)
    return result


def pairwise(left, right, ids):
    """Agreement and discordant correctness between two runs on the same records."""
    agree = sum(left[i]["student"]["prediction"] == right[i]["student"]["prediction"] for i in ids)
    left_only = sum(left[i]["student"]["correct"] and not right[i]["student"]["correct"] for i in ids)
    right_only = sum(right[i]["student"]["correct"] and not left[i]["student"]["correct"] for i in ids)
    both = sum(left[i]["student"]["correct"] and right[i]["student"]["correct"] for i in ids)
    return {"count": len(ids), "prediction_agreement": agree / len(ids), "both_correct": both,
            "neither_correct": len(ids) - both - left_only - right_only,
            "left_only_correct": left_only, "right_only_correct": right_only,
            "mcnemar_exact_p": mcnemar_exact(left_only, right_only)}


def compare(runs):
    ids = join_runs(runs)
    metrics = {name: run_metrics([rows[i] for i in ids]) for name, rows in runs.items()}
    pairs = {f"{a} vs {b}": pairwise(runs[a], runs[b], ids) for a, b in itertools.combinations(runs, 2)}
    return {"count": len(ids), "runs": metrics, "pairs": pairs}


def percent(value):
    return f"{value:.1%}"


def report_markdown(result, title="Public benchmark comparison"):
    names = list(result["runs"])
    lines = [f"# {title}", "",
             f"All runs answered the same {result['count']} records, joined by ID with identical "
             "reference labels. Labels come from the public dataset's human annotation; the subset was "
             "selected by a seeded, label-blind rule that keeps contrastive families together.", "",
             "| Run | Correct | Accuracy | 95% CI | Families complete | ECE ↓ | NLL ↓ | Brier ↓ | Invalid |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        m = result["runs"][name]
        fam = m["family_complete"]
        lines.append(
            f"| {name} | {m['correct']}/{m['count']} | {percent(m['accuracy'])} | "
            f"{percent(m['accuracy_ci95'][0])}–{percent(m['accuracy_ci95'][1])} | "
            f"{fam['complete']}/{fam['families']} | "
            + " | ".join(f"{m[k]:.3f}" if k in m else "N/A"
                         for k in ("ece10", "mean_negative_log_likelihood", "mean_multiclass_brier"))
            + f" | {m['errors']} |")
    lines += ["", "## Per domain", "", "| Domain | " + " | ".join(names) + " |",
              "|---|" + "---:|" * len(names)]
    for domain in sorted(result["runs"][names[0]]["by_domain"]):
        lines.append(f"| {domain} | " + " | ".join(
            f"{result['runs'][n]['by_domain'][domain]['correct']}/{result['runs'][n]['by_domain'][domain]['count']}"
            for n in names) + " |")
    lines += ["", "## Paired correctness", ""]
    for label, pair in result["pairs"].items():
        left, right = label.split(" vs ")
        lines += [f"**{label}**: predictions agree on {percent(pair['prediction_agreement'])} of records. "
                  f"Both correct {pair['both_correct']}, neither {pair['neither_correct']}, "
                  f"only {left} {pair['left_only_correct']}, only {right} {pair['right_only_correct']}. "
                  f"Exact McNemar two-sided p = {pair['mcnemar_exact_p']:.4f}.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="NAME=ROWS_JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Public benchmark comparison")
    args = parser.parse_args()
    runs = {}
    for item in args.runs:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"Expected NAME=path/to/rows.jsonl, got {item!r}")
        if name in runs:
            parser.error(f"Duplicate run name {name!r}")
        runs[name] = load_rows(path)
    result = compare(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")
    (args.output_dir / "REPORT.md").write_text(report_markdown(result, args.title), encoding="utf-8")
    print(json.dumps({name: {k: m[k] for k in ("count", "correct", "accuracy", "errors")}
                      for name, m in result["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
