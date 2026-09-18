"""Aggregate per-dataset public benchmark runs into one suite-level report."""

import argparse
import json
import statistics
from pathlib import Path

from nimble.evaluation.compare_public import (accuracy_block, compare, join_runs, load_rows, pairwise,
                                              percent)
from nimble.evaluation.evaluate_public import distribution_metrics

DEFAULT_ROOT = Path("evaluations/public")
DEFAULT_DATA = Path("data/public")
DEFAULT_RUNS = ("nimble-9b", "jev-1.13.0")
MULTILINGUAL = ("massive-en-US", "massive-de-DE")
MANIFEST_FIELDS = ("dataset", "license", "source_url", "release_year", "count", "families")


def subset_manifest(name, data_root):
    """Provenance fields from the subset's data manifest, when the data directory is present."""
    path = Path(data_root) / name / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {key: manifest.get(key) for key in MANIFEST_FIELDS}


def run_status(subset_dir, run):
    """Return None when a run is complete, else the reason it cannot be aggregated yet."""
    if not (subset_dir / run / "rows.jsonl").exists():
        return f"{run}: no rows.jsonl"
    if not (subset_dir / run / "summary.json").exists():
        return f"{run}: still running (no summary.json)"
    return None


def comparison_for(subset_dir, runs):
    """Compute the subset's comparison from the current rows and write it beside them."""
    rows = {run: load_rows(subset_dir / run / "rows.jsonl") for run in runs}
    result = compare(rows)
    path = subset_dir / "comparison" / "comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result, rows


def row_distribution(row):
    """Per-row divergence from the human distribution; API probabilities are renormalized after rounding."""
    human = row["reference"].get("distribution")
    probabilities = row["student"].get("probabilities")
    if human is None or not probabilities or "error" in row:
        return None
    total = sum(probabilities.values())
    return distribution_metrics({k: v / total for k, v in probabilities.items()}, human)


def distribution_summary(rows):
    """Mean divergence from the human label distribution over records that carry one."""
    values = [d for d in map(row_distribution, rows) if d]
    if not values:
        return None
    return {"count": len(values),
            "mean_jensen_shannon_bits": statistics.mean(v["jensen_shannon_bits"] for v in values),
            "mean_total_variation": statistics.mean(v["total_variation"] for v in values)}


def score_error(rows):
    """Mean absolute expected-score error over valid score records."""
    errors = [r["student"]["absolute_score_error"] for r in rows
              if r["type"] == "score" and "error" not in r and "absolute_score_error" in r["student"]]
    return statistics.mean(errors) if errors else None


def subset_summary(name, manifest, comparison, rows):
    """One suite row: provenance, per-run quality, and the pairwise blocks."""
    ids = sorted(rows[next(iter(rows))])
    kinds = sorted({rows[next(iter(rows))][i]["type"] for i in ids})
    summary = {"subset": name, **manifest, "type": "/".join(kinds), "n": comparison["count"],
               "families": comparison["runs"][next(iter(rows))]["family_complete"]["families"],
               "runs": {}, "pairs": comparison["pairs"]}
    for run, metrics in comparison["runs"].items():
        run_rows = [rows[run][i] for i in ids]
        summary["runs"][run] = {
            "correct": metrics["correct"], "count": metrics["count"], "accuracy": metrics["accuracy"],
            "accuracy_ci95": metrics["accuracy_ci95"], "errors": metrics["errors"],
            "family_complete": metrics["family_complete"]["complete"],
            "family_complete_accuracy": metrics["family_complete"]["accuracy"],
            "ece10": metrics.get("ece10"), "mean_multiclass_brier": metrics.get("mean_multiclass_brier"),
            "mean_negative_log_likelihood": metrics.get("mean_negative_log_likelihood"),
            "mean_absolute_score_error": score_error(run_rows),
            "distribution": distribution_summary(run_rows),
        }
    return summary


def averages(subsets, runs):
    """Macro (unweighted mean of subset accuracies) and micro (pooled) averages, overall and per type."""
    result = {}
    groups = {"all": subsets}
    for kind in sorted({s["type"] for s in subsets}):
        groups[kind] = [s for s in subsets if s["type"] == kind]
    for label, group in groups.items():
        if not group:
            continue
        result[label] = {"subsets": len(group), "runs": {}}
        for run in runs:
            correct = sum(s["runs"][run]["correct"] for s in group)
            count = sum(s["runs"][run]["count"] for s in group)
            result[label]["runs"][run] = {
                "macro_accuracy": statistics.mean(s["runs"][run]["accuracy"] for s in group),
                "micro_accuracy": correct / count, "correct": correct, "count": count}
    return result


def multilingual(root, runs):
    """Paired locale comparison: each run's en-US and de-DE answers joined on identical ids."""
    result = {}
    for run in runs:
        paths = [Path(root) / name / run / "rows.jsonl" for name in MULTILINGUAL]
        if not all(p.exists() and (p.parent / "summary.json").exists() for p in paths):
            return None
        locales = {name: load_rows(path) for name, path in zip(MULTILINGUAL, paths)}
        ids = join_runs(locales)
        first, second = MULTILINGUAL
        result[run] = {name: accuracy_block([locales[name][i] for i in ids]) for name in MULTILINGUAL}
        result[run]["paired"] = pairwise(locales[first], locales[second], ids)
    return result


def summarize(root=DEFAULT_ROOT, runs=DEFAULT_RUNS, data_root=DEFAULT_DATA):
    """Aggregate every complete subset under root; incomplete ones are listed with a reason."""
    root, runs = Path(root), tuple(runs)
    if len(runs) < 2:
        raise ValueError("The suite summary needs at least two runs")
    subsets, skipped = [], {}
    for subset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        reasons = [reason for reason in (run_status(subset_dir, run) for run in runs) if reason]
        if reasons:
            skipped[subset_dir.name] = "; ".join(reasons)
            continue
        try:
            comparison, rows = comparison_for(subset_dir, runs)
        except ValueError as error:
            skipped[subset_dir.name] = str(error)
            continue
        manifest = subset_manifest(subset_dir.name, data_root)
        subsets.append(subset_summary(subset_dir.name, manifest, comparison, rows))
    result = {"root": str(root), "runs": list(runs), "subsets": subsets, "skipped": skipped,
              "records": sum(s["n"] for s in subsets)}
    if subsets:
        result["averages"] = averages(subsets, runs)
    if all(root.joinpath(name).is_dir() for name in MULTILINGUAL):
        try:
            result["multilingual"] = multilingual(root, runs)
        except ValueError as error:
            skipped["multilingual"] = str(error)
    return result


def _num(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _ci(block):
    low, high = block["accuracy_ci95"]
    return f"{percent(block['accuracy'])} ({percent(low)}–{percent(high)})"


def report_markdown(result):
    """Render the suite tables in the style of the per-dataset reports."""
    runs = result["runs"]
    lines = ["# Public benchmark suite", "",
             f"{len(result['subsets'])} subsets and {result['records']} records with results for "
             f"every run ({', '.join(runs)}). Each subset was scored on identical records, joined by ID "
             "with identical human reference labels.", ""]
    if result["subsets"]:
        pair_label = f"{runs[0]} vs {runs[1]}"
        lines += ["## Accuracy", "",
                  "| Subset | Type | n | Families | " + " | ".join(f"{r} accuracy (95% CI)" for r in runs)
                  + " | Agreement | McNemar p |", "|---|---|---:|---:|" + "---:|" * len(runs) + "---:|---:|"]
        for s in result["subsets"]:
            pair = s["pairs"][pair_label]
            lines.append(f"| {s['subset']} | {s['type']} | {s['n']} | {s['families']} | "
                         + " | ".join(_ci(s["runs"][r]) for r in runs)
                         + f" | {percent(pair['prediction_agreement'])} | {pair['mcnemar_exact_p']:.4f} |")
        lines += ["", "## Probabilities and families", "",
                  "| Subset | " + " | ".join(f"{r} families complete" for r in runs) + " | "
                  + " | ".join(f"{r} ECE" for r in runs) + " | " + " | ".join(f"{r} Brier" for r in runs)
                  + " | " + " | ".join(f"{r} score MAE" for r in runs) + " | "
                  + " | ".join(f"{r} JSD / TVD" for r in runs) + " |",
                  "|---|" + "---:|" * (5 * len(runs))]
        for s in result["subsets"]:
            cells = [f"{s['runs'][r]['family_complete']}/{s['families']}" for r in runs]
            cells += [_num(s["runs"][r]["ece10"]) for r in runs]
            cells += [_num(s["runs"][r]["mean_multiclass_brier"]) for r in runs]
            cells += [_num(s["runs"][r]["mean_absolute_score_error"]) for r in runs]
            for r in runs:
                d = s["runs"][r]["distribution"]
                cells.append("—" if d is None else
                             f"{d['mean_jensen_shannon_bits']:.3f} / {d['mean_total_variation']:.3f}")
            lines.append(f"| {s['subset']} | " + " | ".join(cells) + " |")
        lines += ["", "## Averages", "",
                  "| Group | Subsets | " + " | ".join(f"{r} macro | {r} micro" for r in runs) + " |",
                  "|---|---:|" + "---:|---:|" * len(runs)]
        for label, group in result["averages"].items():
            lines.append(f"| {label} | {group['subsets']} | " + " | ".join(
                f"{percent(group['runs'][r]['macro_accuracy'])} | {percent(group['runs'][r]['micro_accuracy'])}"
                for r in runs) + " |")
        lines += ["", "Macro averages weight every subset equally; micro averages pool all records, so "
                  "larger subsets count for more.", ""]
    block = result.get("multilingual")
    if block:
        first, second = MULTILINGUAL
        lines += ["## Paired multilingual: MASSIVE en-US and de-DE on identical ids", "",
                  f"| Run | {first} | {second} | Locale agreement | Right in both | Right only {first} | "
                  f"Right only {second} | McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for run, m in block.items():
            p = m["paired"]
            lines.append(f"| {run} | {_ci(m[first])} | {_ci(m[second])} | "
                         f"{percent(p['prediction_agreement'])} | {p['both_correct']} | "
                         f"{p['left_only_correct']} | {p['right_only_correct']} | {p['mcnemar_exact_p']:.4f} |")
        lines.append("")
    if result["skipped"]:
        lines += ["## Not yet aggregated", ""]
        lines += [f"- {name}: {reason}" for name, reason in result["skipped"].items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    args = parser.parse_args()
    result = summarize(args.root, args.runs, args.data_root)
    (args.root / "suite.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.root / "SUITE.md").write_text(report_markdown(result), encoding="utf-8")
    print(json.dumps({"subsets": len(result["subsets"]), "records": result["records"],
                      "skipped": result["skipped"],
                      "averages": {k: {r: round(v["macro_accuracy"], 4) for r, v in g["runs"].items()}
                                   for k, g in result.get("averages", {}).items()}}, indent=2))


if __name__ == "__main__":
    main()
