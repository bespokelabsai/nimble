"""Score public benchmark records with the TypeSafe Jev API; references are joined only afterwards."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from nimble.datasets.dataset_io import canonical, request_for, validate_teacher
from nimble.datasets.public_records import reference_keys
from nimble.evaluation.evaluate_pilot import assess, summarize, target_key
from nimble.evaluation.evaluate_public import BINS, expected_calibration_error
from nimble.paths import PROJECT_ROOT

API_URL = "https://api.typesafe.ai/v1/systemone"
API_KEY_VARIABLE = "TYPESAFE_API_KEY"
DEFAULT_MODEL = "jev-1.13.0"
RETRYABLE_STATUSES = (429, 500, 502, 503, 504, 529)
FORBIDDEN_KEYS = {"reference", "teacher", "teacher_targets"}


class ApiError(Exception):
    """An HTTP failure whose status decides whether it is retried, recorded, or fatal."""

    def __init__(self, status):
        super().__init__(f"TypeSafe HTTP {status}")
        self.status = status


def load_api_key(dotenv=PROJECT_ROOT / ".env"):
    """Read the key from the environment, loading .env the way the curation scripts do; never log it."""
    if not os.environ.get(API_KEY_VARIABLE) and Path(dotenv).exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv, override=False)
    value = os.environ.get(API_KEY_VARIABLE)
    if not value:
        raise ValueError(f"{API_KEY_VARIABLE} is not set in the environment or {dotenv}")
    return value


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def request_payload(row, model):
    """Build the API request from the model-visible input only, and prove nothing else leaked."""
    if list(row["input"]["questions"]) != ["decision"]:
        raise ValueError(f"{row['id']}: exactly one decision question is required")
    payload = request_for(row, model)
    leaked = reference_keys(payload) & FORBIDDEN_KEYS
    if leaked:
        raise ValueError(f"Reference-side keys reached the request: {sorted(leaked)}")
    return payload


def http_transport(api_key, url=API_URL, timeout=90):
    """Return a callable posting one payload; the key lives only inside this closure."""
    def call(payload):
        request = urllib.request.Request(
            url, data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Never expose provider bodies or headers.
            raise ApiError(exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiError(0) from None
    return call


def request_with_retry(transport, payload, attempts=5, sleep=time.sleep, base_delay=1.0):
    """Retry rate limits, server errors, and connection failures with exponential backoff."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return transport(payload)
        except ApiError as exc:
            if exc.status in (401, 403):
                raise
            if exc.status not in RETRYABLE_STATUSES and exc.status != 0:
                raise
            if attempt == attempts - 1:
                raise
            sleep(base_delay * 2 ** attempt)


def api_probabilities(row, answer):
    """Normalize the API distribution over the record's candidates, preserving criterion order."""
    question = row["input"]["questions"]["decision"]
    kind = question["type"]
    if kind == "noul":
        return {"false": 1 - answer["noul"], "true": answer["noul"]}
    labels = (list(question["criteria"]) if kind == "choice"
              else [str(i) for i in range(len(question["criteria"]))])
    total = sum(answer["probabilities"].values())
    if not total > 0:
        raise ValueError("API probabilities sum to zero")
    return {label: answer["probabilities"][label] / total for label in labels}


def base_row(row, elapsed_seconds):
    """The record fields every saved row carries, in evaluate_public's layout."""
    return {"id": row["id"], "type": row["input"]["questions"]["decision"]["type"], "domain": row["domain"],
            "family": row["family"], "split": row.get("split", "test"), "reference": row["reference"],
            "input_sha256": fingerprint(row["input"]), "elapsed_seconds": elapsed_seconds}


def jev_row(row, response, elapsed_seconds):
    """Join the reference after inference."""
    kind = row["input"]["questions"]["decision"]["type"]
    answer = response["answers"]["decision"]
    probabilities = api_probabilities(row, answer)
    student = assess(probabilities, row["reference"]["target"], kind)
    if kind == "choice":
        prediction = answer["choice"]
    elif kind == "noul":
        prediction = answer["noul"] > 0.5
    else:
        prediction = student["prediction"]
    student.update(prediction=prediction, correct=prediction == row["reference"]["target"])
    return {**base_row(row, elapsed_seconds), "student": student,
            "raw_student": {"model": response["model"], "response": response}}


def error_row(row, error, elapsed_seconds):
    return {**base_row(row, elapsed_seconds), "student": {"prediction": None, "correct": False}, "error": error}


def validate_reference(row):
    """Reject a malformed reference before any billed request; API problems are recorded per row instead."""
    question = row["input"]["questions"]["decision"]
    try:
        key = target_key(row["reference"]["target"], question["type"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{row['id']}: {exc}") from None
    if question["type"] == "choice" and key not in question["criteria"]:
        raise ValueError(f"{row['id']}: reference target {key!r} is not one of the criteria")
    if question["type"] == "score" and not 0 <= int(key) < len(question["criteria"]):
        raise ValueError(f"{row['id']}: reference level {key} is outside the rubric")


def score_record(row, model, transport, sleep=time.sleep, attempts=5):
    """Request one decision; invalid or failed responses become recorded errors, not crashes."""
    payload = request_payload(row, model)
    started = time.perf_counter()
    try:
        response = request_with_retry(transport, payload, attempts=attempts, sleep=sleep)
        validate_teacher(row, response, model)
        return jev_row(row, response, time.perf_counter() - started)
    except ApiError as exc:
        if exc.status in (401, 403):
            raise RuntimeError("TypeSafe rejected the credentials; aborting the run") from None
        return error_row(row, str(exc), time.perf_counter() - started)
    except (ValueError, KeyError, TypeError) as exc:
        return error_row(row, f"Invalid response: {type(exc).__name__}: {exc}",
                         time.perf_counter() - started)


def group_summary(rows):
    """Accuracy over every row, with probabilistic means over the rows that returned a valid answer."""
    valid = [r for r in rows if "error" not in r]
    means = summarize(valid, "student") if valid else {}
    result = {}
    for kind in ("all", "choice", "noul", "score"):
        selected = [r for r in rows if kind == "all" or r["type"] == kind]
        if not selected:
            continue
        group = {"count": len(selected), "valid": sum("error" not in r for r in selected),
                 "correct": sum(r["student"]["correct"] for r in selected)}
        group["accuracy"] = group["correct"] / len(selected)
        for key, value in means.get(kind, {}).items():
            if key.startswith("mean_"):
                group[key] = value
        result[kind] = group
    return result


def summarize_rows(rows):
    """Mirror evaluate_public.summarize_rows while counting invalid responses as incorrect."""
    valid = [r for r in rows if "error" not in r]
    times = sorted(r["elapsed_seconds"] for r in rows)
    report = {"count": len(rows), "valid": len(valid), "errors": len(rows) - len(valid),
              "summary": group_summary(rows),
              "calibration": {"ece10": expected_calibration_error(
                  (r["student"]["top_probability"], r["student"]["correct"]) for r in valid) if valid else None,
                  "bins": BINS, "calibration_n": len(valid),
                  "confidence": "top API probability after renormalization; invalid responses excluded"},
              "by_domain": {d: group_summary([r for r in rows if r["domain"] == d])
                            for d in sorted({r["domain"] for r in rows})},
              "timing": {"total_request_seconds": sum(times), "median_seconds": statistics.median(times),
                         "p95_seconds": times[math.ceil(.95 * len(times)) - 1],
                         "boundary": "client wall time per request including retries and HTTPS transport"},
              "usage": {key: sum(r["raw_student"]["response"]["usage"][key] for r in valid)
                        for key in ("input_tokens", "output_tokens")}}
    return report


def read_saved(path, lookup):
    rows = []
    if path.exists() and path.stat().st_size:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] not in lookup or row["input_sha256"] != fingerprint(lookup[row["id"]]["input"]):
                raise ValueError(f"Saved row {row.get('id')} no longer matches the dataset")
            rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Saved rows contain duplicate IDs")
    return rows


def run(data, output, model=DEFAULT_MODEL, concurrency=4, limit=None, transport=None,
        sleep=time.sleep, attempts=5, progress=None):
    """Score records, resuming a partial run, and write rows, a summary, and a provenance manifest."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    data, output = Path(data), Path(output)
    raw = data.read_bytes()
    records = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset must be nonempty with unique IDs")
    if any(FORBIDDEN_KEYS & set(r["input"]) or "teacher" in r for r in records):
        raise ValueError("Public benchmark records must not carry teacher annotations")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        records = records[:limit]
    for r in records:
        validate_reference(r)
    lookup = {r["id"]: r for r in records}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": str(data.resolve()), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model": model, "api_url": API_URL, "concurrency": concurrency, "limit": limit,
                "retry_attempts": attempts, "started_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(saved.get(k) != manifest[k] for k in ("dataset_sha256", "model", "api_url", "limit")):
            raise ValueError("Existing run in this directory used different data or model settings")
        manifest = saved
    rows_path = output / "rows.jsonl"
    rows = read_saved(rows_path, lookup)
    retried = sum("error" in r for r in rows)
    if retried:
        # A failed request is not a result: drop it so the resumed run asks again.
        rows = [r for r in rows if "error" not in r]
        rows_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
                             encoding="utf-8")
    done = {r["id"] for r in rows}
    pending = [r for r in records if r["id"] not in done]
    if pending:
        if transport is None:
            transport = http_transport(load_api_key())
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool, rows_path.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(score_record, r, model, transport, sleep, attempts) for r in pending]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if progress is not None and (len(rows) % 10 == 0 or len(rows) == len(records)):
                    progress(len(rows), len(records))
    order = {r["id"]: i for i, r in enumerate(records)}
    rows.sort(key=lambda r: order[r["id"]])
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(), done=len(rows),
                    errors=sum("error" in r for r in rows), resumed=bool(done), retried_errors=retried)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {**manifest, **summarize_rows(rows)}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL of public benchmark records")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run(args.data, args.output_dir, model=args.model, concurrency=args.concurrency,
                 limit=args.limit, progress=lambda done, total: print(f"{done}/{total}", flush=True))
    print(json.dumps({k: report[k] for k in ("count", "valid", "errors", "summary", "calibration", "usage")},
                     indent=2))


if __name__ == "__main__":
    main()
