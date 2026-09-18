"""Convert public, human-labeled datasets into this repo's evaluation record contract.

The retained holdout is synthetic and model-checked. These converters add external
benchmarks whose labels were produced by people, so the scorer can be measured
against annotations it had no part in creating.

Each dataset lives in its own module under nimble.datasets.public_sources and is
discovered here. A module defines NAME, SOURCE_URL, LICENSE, NOTE, RELEASE_YEAR,
SUBSETS, rows(path, subset) over a downloaded local file, and record(raw, subset)
returning a record built with public_records.record_for, or None to skip a row. Conversion needs no network access or dataset library. See
docs/PUBLIC_BENCHMARKS.md for the download steps.
"""

import argparse
import hashlib
import importlib
import json
import pkgutil
import random
from collections import Counter
from pathlib import Path

from nimble.datasets import public_sources
from nimble.datasets.public_records import source_digest
from nimble.evaluation.evaluate_pilot import adapt_input

REQUIRED_ATTRIBUTES = ("NAME", "SOURCE_URL", "LICENSE", "NOTE", "RELEASE_YEAR", "SUBSETS")


def validate_source(module, expected_name=None):
    """Check that a source module exposes the full contract; name the missing piece."""
    name = getattr(module, "__name__", repr(module))
    for attribute in REQUIRED_ATTRIBUTES:
        if not hasattr(module, attribute):
            raise ValueError(f"{name} is missing {attribute}")
    for function in ("rows", "record"):
        if not callable(getattr(module, function, None)):
            raise ValueError(f"{name} is missing a callable {function}")
    if expected_name is not None and module.NAME != expected_name:
        raise ValueError(f"{name}.NAME is {module.NAME!r} but the module is named {expected_name!r}")
    return module


def load_sources(package=public_sources):
    """Import every module in the sources package and validate its contract."""
    sources = {}
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        validate_source(module, info.name)
        sources[module.NAME] = module
    return sources


def jsonl_line(record):
    """Serialize one record as a line that survives readers using str.splitlines().

    ensure_ascii=False keeps text readable, but leaves U+2028, U+2029, and U+0085
    raw, and splitlines() treats those as line breaks. Escape only those three;
    the parsed value is identical.
    """
    text = json.dumps(record, ensure_ascii=False, allow_nan=False)
    for separator in (" ", " ", ""):
        text = text.replace(separator, "\\u%04x" % ord(separator))
    return text + "\n"


def select(records, seed, limit):
    """Take whole families at random using only metadata, never the reference values."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when given")
    families = {}
    for record in records:
        families.setdefault(record["family"], []).append(record)
    strata = {}
    for family, group in families.items():
        strata.setdefault(group[0]["domain"], []).append(family)
    rng = random.Random(seed)
    order = []
    for domain in sorted(strata):
        keys = sorted(strata[domain])
        rng.shuffle(keys)
        order.append(keys)
    selected, index, stop = [], 0, False
    while not stop and any(index < len(keys) for keys in order):
        for keys in order:
            if index >= len(keys):
                continue
            group = families[keys[index]]
            # Whole families only, so contrastive siblings are never split across the boundary.
            if limit is not None and len(selected) + len(group) > limit:
                stop = True
                break
            selected.extend(group)
        index += 1
    selected.sort(key=lambda record: record["id"])
    return selected


def select_ids(records, ids):
    """Take exactly the listed ids, using only ids and family metadata; families stay whole."""
    wanted = set(ids)
    if len(wanted) != len(ids):
        raise ValueError("ids-from manifest lists duplicate ids")
    by_id = {record["id"]: record for record in records}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"{len(missing)} ids from the manifest are absent after conversion, e.g. {missing[:3]}")
    families = {}
    for record in records:
        families.setdefault(record["family"], []).append(record["id"])
    for identifier in ids:
        members = families[by_id[identifier]["family"]]
        absent = sorted(set(members) - wanted)
        if absent:
            raise ValueError(f"family {by_id[identifier]['family']!r} is only partially covered; "
                             f"missing {absent[:3]}")
    return sorted((by_id[identifier] for identifier in wanted), key=lambda record: record["id"])


def manifest_text(manifest):
    """Pretty-print the manifest with the id list on one line, so the file diffs by field."""
    head = json.dumps({k: v for k, v in manifest.items() if k != "ids"}, indent=2, ensure_ascii=False)
    ids = json.dumps(manifest["ids"], ensure_ascii=False, separators=(",", ":"))
    return head[:-2] + ',\n  "ids": ' + ids + "\n}\n"


def build(name, rows, output, seed=20260918, limit=None, subset="", measure=None, max_input_tokens=2048,
          source=None, source_sha256=None, ids_from=None):
    """Convert, length-filter, and select records, then write the payload and its manifest."""
    sources = load_sources()
    if name not in sources:
        raise ValueError(f"Unknown dataset: {name}")
    spec = sources[name]
    if spec.SUBSETS and subset not in spec.SUBSETS:
        raise ValueError(f"{name} requires --subset, one of {list(spec.SUBSETS)}")
    if not spec.SUBSETS and subset:
        raise ValueError(f"{name} does not take a subset")
    if ids_from is not None and limit is not None:
        raise ValueError("ids-from and limit are mutually exclusive")
    converted, dropped, skipped = [], 0, 0
    for raw in rows:
        record = spec.record(raw, subset)
        if record is None:
            skipped += 1
            continue
        if measure is not None:
            context, schema = adapt_input(record["input"])
            if measure(context, schema) > max_input_tokens:
                dropped += 1
                continue
        converted.append(record)
    if len({record["id"] for record in converted}) != len(converted):
        raise ValueError("Converted records must have unique IDs")
    if not converted:
        raise ValueError("No records survived conversion and length filtering")
    ids_manifest = None
    if ids_from is not None:
        ids_manifest = json.loads(Path(ids_from).read_text(encoding="utf-8"))
        selected = select_ids(converted, ids_manifest["ids"])
        selection = (f"Exactly the ids listed in {Path(ids_from).name}, matched by id with whole families;"
                     " reference targets and distributions are never inspected.")
    else:
        selected = select(converted, seed, limit)
        selection = ("Whole families drawn without replacement, round-robin across domain strata. Only family"
                     " and domain metadata determine the order; reference targets and distributions are never"
                     " inspected. Contrastive siblings are always kept together.")
    if not selected:
        raise ValueError("No records selected: " + ("the ids-from manifest lists no ids" if ids_from is not None
                                                   else f"no whole family fits within --limit {limit}"))
    payload = "".join(jsonl_line(record) for record in selected).encode()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "all.jsonl"
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Existing dataset differs; choose a new directory")
    path.write_bytes(payload)
    manifest = {
        "dataset": name, "subset": subset, "source_url": spec.SOURCE_URL, "license": spec.LICENSE,
        "note": spec.NOTE, "release_year": spec.RELEASE_YEAR,
        "source": str(source) if source is not None else None,
        "source_sha256": source_sha256, "dataset_sha256": hashlib.sha256(payload).hexdigest(), "seed": seed,
        "count": len(selected), "families": len({record["family"] for record in selected}),
        "limit": limit, "converted": len(converted), "skipped": skipped, "dropped_over_length": dropped,
        "max_input_tokens": max_input_tokens if measure is not None else None,
        "length_filter": "applied with the serving prompt builder" if measure is not None
                         else "not applied; no tokenizer was supplied",
        "selection": selection,
        "ids_from": None if ids_manifest is None else {
            "path": str(ids_from), "dataset": ids_manifest.get("dataset"), "subset": ids_manifest.get("subset"),
            "dataset_sha256": ids_manifest.get("dataset_sha256")},
        "types": dict(Counter(record["input"]["questions"]["decision"]["type"] for record in selected)),
        "labels": dict(Counter(str(record["reference"]["target"]) for record in selected)),
        "domains": dict(Counter(record["domain"] for record in selected)),
        "human_reviewed": True,
        "ids": [record["id"] for record in selected],
    }
    (output / "manifest.json").write_text(manifest_text(manifest), encoding="utf-8")
    return manifest


def main():
    sources = load_sources()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(sources))
    parser.add_argument("--source", type=Path, required=True, help="Downloaded upstream file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", default="", help="Subset name for datasets that define SUBSETS")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids-from", type=Path,
                        help="Manifest of another build; select exactly its ids instead of sampling")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer", help="Local tokenizer path; without it no length filter is applied")
    args = parser.parse_args()
    measure = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        from nimble.scoring.parallel_schema import prepare_prompts

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

        def measure(context, schema):
            # Measure the exact serving prompt, so the budget matches what the scorer rejects.
            prepared = prepare_prompts(tokenizer, context, schema, max_input_tokens=10**9)
            return max(map(len, prepared.full_ids))

    try:
        spec = sources[args.dataset]
        manifest = build(args.dataset, spec.rows(args.source, args.subset), args.output_dir, args.seed,
                         args.limit, args.subset, measure, args.max_input_tokens,
                         source=args.source, source_sha256=source_digest(args.source), ids_from=args.ids_from)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({k: v for k, v in manifest.items() if k != "ids"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
