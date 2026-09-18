"""Evaluate public, human-labeled benchmark records without any teacher annotation."""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from nimble.evaluation.evaluate_pilot import adapt_input, assess, summarize

BINS = 10
DISTRIBUTION_METRICS = ("jensen_shannon_bits", "total_variation", "human_entropy_bits")
SCORER_OPTIONS = ("device_map", "max_gpu_memory", "dtype", "attention")
RESUME_INVARIANTS = ("dataset_sha256", "model", "revision", "model_path", "backend", "temperature",
                     "max_input_tokens", "mode", "shuffle_seed") + SCORER_OPTIONS


def package_versions(names):
    """Record interpreter and backend versions without failing on an absent package."""
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def entropy(distribution):
    """Shannon entropy in bits; 0 log 0 is treated as 0."""
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)


def total_variation(left, right):
    """Total variation distance, bounded in [0, 1]."""
    return sum(abs(left[k] - right[k]) for k in left) / 2


def jensen_shannon(left, right):
    """Jensen-Shannon divergence in bits, bounded in [0, 1] for base-2 logarithms."""
    total = 0.0
    for key in left:
        p, q = left[key], right[key]
        mean = (p + q) / 2
        if p > 0:
            total += p * math.log2(p / mean) / 2
        if q > 0:
            total += q * math.log2(q / mean) / 2
    return min(max(total, 0.0), 1.0)  # clamp floating-point overshoot of the analytic bound


def distribution_metrics(probabilities, human):
    """Compare model candidate probabilities with a human label distribution over the same options.

    Both inputs are already validated: assess() checks the model probabilities and
    record_for() checks a dataset's human distribution when the record is built.
    """
    if set(probabilities) != set(human):
        raise ValueError("Model candidates and the human distribution cover different options")
    return {"jensen_shannon_bits": jensen_shannon(probabilities, human),
            "total_variation": total_variation(probabilities, human),
            "human_entropy_bits": entropy(human)}


def expected_calibration_error(pairs, bins=BINS):
    """Equal-width ECE over (confidence, correct) pairs, matching evaluate_banking77."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Calibration needs at least one scored record")
    p = [float(confidence) for confidence, _ in pairs]
    y = [int(bool(correct)) for _, correct in pairs]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in p):
        raise ValueError("Confidence values must be finite and in [0, 1]")
    error = 0.0
    for i in range(bins):
        indexes = [j for j, x in enumerate(p)
                   if i / bins <= x and (x < (i + 1) / bins or i == bins - 1 and x <= 1)]
        if indexes:
            error += abs(sum(p[j] - y[j] for j in indexes)) / len(p)
    return error


def model_input(row):
    """Return only the model-visible context and schema; references stay out by construction."""
    questions = row["input"]["questions"]
    if list(questions) != ["decision"]:
        raise ValueError(f"{row['id']}: exactly one decision field is required")
    # adapt_input forwards noul criteria unchanged, so they must be keyed by candidate value.
    if questions["decision"]["type"] == "noul" and set(questions["decision"]["criteria"]) != {"false", "true"}:
        raise ValueError(f"{row['id']}: noul criteria must be keyed 'false' and 'true'")
    return adapt_input(row["input"])


def shuffled_schema(schema, row_id, shuffle_seed):
    """Permute candidate order per record so A-Z position bias can be measured."""
    schema = json.loads(json.dumps(schema))
    random.Random(f"{shuffle_seed}:{row_id}").shuffle(schema["decision"]["choices"])
    return schema


def evaluate_records(records, scorer, shuffle_seed=None, mode="independent", progress=None, on_row=None):
    """Score every record with an already-loaded scorer; no teacher annotation is consulted."""
    if not records:
        raise ValueError("Dataset must contain at least one record")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset must have unique IDs")
    rows = []
    for row in records:
        context, schema = model_input(row)
        kind = row["input"]["questions"]["decision"]["type"]
        if shuffle_seed is not None:
            schema = shuffled_schema(schema, row["id"], shuffle_seed)
        started = time.perf_counter()
        prediction = scorer.score(context, schema, mode=mode)
        seconds = time.perf_counter() - started
        field = prediction["fields"]["decision"]
        reference = row["reference"]
        student = assess(field["scores"], reference["target"], kind)
        result = {"id": row["id"], "type": kind, "domain": row["domain"], "family": row["family"],
                  "split": row.get("split", "test"), "reference": reference, "student": student,
                  "raw_student": prediction, "elapsed_seconds": seconds}
        if "distribution" in reference:
            result["distribution"] = distribution_metrics(student["probabilities"], reference["distribution"])
        rows.append(result)
        if on_row is not None:
            on_row(result)
        if progress is not None and (len(rows) % 10 == 0 or len(rows) == len(records)):
            progress(len(rows), len(records))
    return rows


def code_counts(rows):
    """Count which letter code was selected, to expose position bias in the report."""
    counts = Counter()
    for row in rows:
        field = row["raw_student"]["fields"]["decision"]
        mapping = field.get("code_to_choice")
        if not mapping:
            return {}
        counts.update([next(code for code, value in mapping.items() if value == field["value"])])
    return dict(sorted(counts.items()))


def summarize_rows(rows):
    """Aggregate accuracy, calibration, and distributional agreement across scored records."""
    times = sorted(r["elapsed_seconds"] for r in rows)
    report = {
        "count": len(rows),
        "summary": summarize(rows, "student"),
        "calibration": {"ece10": expected_calibration_error(
            (r["student"]["top_probability"], r["student"]["correct"]) for r in rows), "bins": BINS,
            "confidence": "top candidate probability; no temperature has been fitted"},
        "selected_code_counts": code_counts(rows),
        "by_domain": {domain: summarize([r for r in rows if r["domain"] == domain], "student")
                      for domain in sorted({r["domain"] for r in rows})},
        "timing": {"total_scoring_seconds": sum(times), "median_seconds": statistics.median(times),
                   "p95_seconds": times[math.ceil(.95 * len(times)) - 1]},
    }
    scored = [r for r in rows if "distribution" in r]
    if scored:
        report["distribution_agreement"] = {
            "count": len(scored),
            **{"mean_" + metric: statistics.mean(r["distribution"][metric] for r in scored)
               for metric in DISTRIBUTION_METRICS},
            "note": "Base-2 Jensen-Shannon and total variation against the human label distribution",
        }
    return report


def build_scorer(backend, model_path, model_id, revision, temperature, max_input_tokens,
                 device_map=None, max_gpu_memory=None, dtype=None, attention=None):
    """Import a backend only when it is actually used, so offline analysis needs no GPU stack."""
    options = {"device_map": device_map, "max_gpu_memory": max_gpu_memory, "dtype": dtype,
               "attention": attention}
    if backend == "mlx":
        if any(value is not None for value in options.values()):
            raise ValueError("device_map, max_gpu_memory, dtype and attention apply to the cuda backend only")
        from nimble.scoring.parallel_scorer import ParallelScorer
        return ParallelScorer(model_path=model_path, model_id=model_id, revision=revision,
                              temperature=temperature, max_input_tokens=max_input_tokens)
    if backend == "cuda":
        from nimble.scoring.cuda_scorer import CudaCandidateScorer
        return CudaCandidateScorer(model_path=model_path, model_id=model_id, revision=revision,
                                   temperature=temperature, max_input_tokens=max_input_tokens,
                                   **{key: value for key, value in options.items() if value is not None})
    raise ValueError("Unknown backend")


def completed_rows(rows_path):
    """Read rows saved by an interrupted run, discarding only a torn, unterminated final line."""
    lines = rows_path.read_text(encoding="utf-8").split("\n")
    torn = lines[-1] != ""
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if torn and number == len(lines):
                break
            raise ValueError(f"{rows_path.name} line {number} is not valid JSON")
    if torn:
        rows_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
                             encoding="utf-8")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{rows_path.name} contains duplicate record IDs")
    return rows


def run(data, output, model_path=None, model_id=None, revision=None, backend="mlx",
        shuffle_seed=None, temperature=1.0, max_input_tokens=2048, mode="independent",
        scorer_factory=None, device_map=None, max_gpu_memory=None, dtype=None, attention=None,
        resume=False):
    """Evaluate a public benchmark file and write rows, a summary, and a provenance manifest.

    Rows are appended as they are scored, and the manifest is written before scoring
    starts, so an interrupted run can continue with resume=True in the same directory.
    """
    data, output = Path(data), Path(output)
    raw = data.read_bytes()
    records = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
    if any("teacher" in r for r in records):
        raise ValueError("Public benchmark records must not carry teacher annotations")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset must have unique IDs")
    output.mkdir(parents=True, exist_ok=True)
    rows_path, summary_path, manifest_path = output / "rows.jsonl", output / "summary.json", output / "manifest.json"
    packages = {"mlx": ("mlx", "mlx-lm", "transformers"),
                "cuda": ("torch", "transformers", "accelerate")}.get(backend, ()) if scorer_factory is None else ()
    manifest = {
        "dataset": str(data.resolve()), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": model_id, "revision": revision, "model_path": str(Path(model_path).resolve()) if model_path else None,
        "backend": backend, "dtype": dtype or "bfloat16", "candidate_projection": "fp32",
        "device_map": device_map, "max_gpu_memory": max_gpu_memory, "attention": attention,
        "temperature": temperature, "temperature_fitted": False,
        "max_input_tokens": max_input_tokens, "mode": mode, "shuffle_seed": shuffle_seed,
        "versions": package_versions(packages),
    }
    previous = []
    if resume:
        if summary_path.exists():
            raise ValueError("summary.json already exists; this run completed and cannot be resumed")
        if not manifest_path.exists() or not rows_path.exists():
            raise ValueError("Nothing to resume: the output directory holds no manifest and rows")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = [key for key in RESUME_INVARIANTS if saved.get(key) != manifest[key]]
        if mismatched:
            raise ValueError(f"Cannot resume: {', '.join(mismatched)} differ from the saved run")
        previous = completed_rows(rows_path)
        unknown = sorted({r["id"] for r in previous} - {r["id"] for r in records})
        if unknown:
            raise ValueError(f"Saved rows are not in the dataset: {unknown[:5]}")
    else:
        for path in (rows_path, summary_path, manifest_path):
            if path.exists():
                raise ValueError(f"{path.name} already exists; use a fresh output directory or resume")
    done = {r["id"] for r in previous}
    pending = [r for r in records if r["id"] not in done]
    started = time.perf_counter()
    scorer = scorer_factory() if scorer_factory is not None else build_scorer(
        backend, model_path, model_id, revision, temperature, max_input_tokens,
        device_map=device_map, max_gpu_memory=max_gpu_memory, dtype=dtype, attention=attention)
    load_seconds = time.perf_counter() - started
    manifest["runtime"] = getattr(scorer, "runtime", None)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with rows_path.open("a", encoding="utf-8") as stream:
        def save(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        new_rows = evaluate_records(
            pending, scorer, shuffle_seed=shuffle_seed, mode=mode, on_row=save,
            progress=lambda done, total: print(f"{len(previous) + done}/{len(records)}", flush=True),
        ) if pending else []
    order = {r["id"]: index for index, r in enumerate(records)}
    rows = sorted(previous + new_rows, key=lambda r: order[r["id"]])
    report = {**manifest, "created_utc": datetime.now(timezone.utc).isoformat(),
              "load_seconds": load_seconds, "resumed_rows": len(previous), **summarize_rows(rows)}
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL of public benchmark records")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--backend", choices=["mlx", "cuda"], default="mlx")
    parser.add_argument("--shuffle-seed", type=int, help="Permute candidate order to measure position bias")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device-map", help="cuda backend accelerate placement, for example auto; "
                        "weights beyond the GPU budget stream from host memory")
    parser.add_argument("--max-gpu-memory", help="GPU weight budget under --device-map, for example 13GiB")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        help="cuda backend load dtype; the default is bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "eager"], help="cuda backend attention implementation")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run in the same output directory")
    args = parser.parse_args()
    report = run(args.data, args.output_dir, model_path=args.model_path, model_id=args.model_id,
                 revision=args.revision, backend=args.backend, shuffle_seed=args.shuffle_seed,
                 temperature=args.temperature, max_input_tokens=args.max_input_tokens,
                 device_map=args.device_map, max_gpu_memory=args.max_gpu_memory, dtype=args.dtype,
                 attention=args.attention, resume=args.resume)
    print(json.dumps({k: report[k] for k in ("count", "summary", "calibration", "selected_code_counts")
                      if k in report}, indent=2))


if __name__ == "__main__":
    main()
