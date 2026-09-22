"""MLX shared-prefix, batched-field classification with a candidate-only LM head."""

import argparse
import math
import sys
import time
from pathlib import Path

from nimble.paths import PROJECT_ROOT

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache

from nimble.scoring.calibration import fitted_temperature
from nimble.scoring.parallel_schema import MODEL_ID, REVISION, choice_key, parse_schema, prepare_prompts


def broadcast_cache(prefix_cache, batch_size):
    """Fork all hybrid states. Branch updates must never mutate the prefix objects.

    broadcast_to can share initial storage; stock cache append operations may
    materialize batch-sized arrays. This is not an allocation-free cache.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    result = []
    for source in prefix_cache:
        if isinstance(source, KVCache):
            target = KVCache()
            target.state = tuple(mx.broadcast_to(x, (batch_size,) + x.shape[1:]) for x in source.state)
        elif isinstance(source, ArraysCache):
            if source.left_padding is not None or source.lengths is not None:
                raise ValueError("Only unpadded, completed prefix states can be broadcast.")
            target = ArraysCache(len(source.state))
            target.state = [None if x is None else mx.broadcast_to(x, (batch_size,) + x.shape[1:])
                            for x in source.state]
        else:
            raise TypeError(f"Unsupported cache: {type(source).__name__}")
        result.append(target)
    return result


def candidate_projection(hidden, weight, token_ids):
    """Compute only selected vocabulary rows, accumulating the small head in FP32."""
    selected_weight = weight[mx.array(token_ids)].astype(mx.float32)
    return hidden.astype(mx.float32) @ selected_weight.T


class ParallelScorer:
    def __init__(self, model_path=None, max_input_tokens=4096, temperature=None,
                 model_id=MODEL_ID, revision=REVISION):
        if not isinstance(max_input_tokens, int) or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer.")
        if temperature is None:
            # The checkpoint's fitted temperature when it has one, else raw probabilities.
            temperature = fitted_temperature(model_id, revision) or 1.0
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        if not mx.metal.is_available():
            raise RuntimeError("MLX requires native Metal GPU access for this runner.")
        mx.set_default_device(mx.gpu)
        path = Path(model_path) if model_path else (
            PROJECT_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots" / REVISION
        )
        if not path.is_dir():
            raise FileNotFoundError(f"Local checkpoint not found: {path}")
        self.model_id, self.revision = model_id, revision
        print(f"Loading {model_id} in MLX...", file=sys.stderr, flush=True)
        self.model, self.tokenizer = load(str(path))
        self.model.eval()
        language_model = getattr(self.model, "language_model", self.model)
        if self.model.model_type not in {"qwen3_5", "gemma3_text"}:
            raise ValueError("Supported architectures are Qwen3.5 and Gemma 3 text.")
        self.backbone = language_model.model
        self.head_weight = (language_model.lm_head.weight if hasattr(language_model, "lm_head")
                            else self.backbone.embed_tokens.weight)
        self.system_role = self.model.model_type != "gemma3_text"
        # Gemma's rotating caches are not covered by the hybrid-cache broadcaster.
        self.default_mode = "parallel" if self.system_role else "independent"
        if self.head_weight.dtype not in (mx.float32, mx.float16, mx.bfloat16):
            raise ValueError("Candidate projection currently requires unquantized weights.")
        self.max_input_tokens = min(max_input_tokens, language_model.args.max_position_embeddings - 1)
        self.temperature = temperature

    def prepare(self, context, schema):
        return prepare_prompts(self.tokenizer, context, schema, self.max_input_tokens,
                               system_role=getattr(self, "system_role", True))

    def evaluate(self, prepared, mode="parallel", field_batch_size=None):
        """Return candidate logits and timings; all modes use identical full prompts.

        parallel: one prefill, batched suffixes; cached_serial: one prefill,
        independent suffixes; independent: recompute each entire prompt.
        """
        if mode not in ("parallel", "cached_serial", "independent"):
            raise ValueError("Unknown evaluation mode.")
        count = len(prepared.names)
        if field_batch_size is not None and (not isinstance(field_batch_size, int) or field_batch_size < 1):
            raise ValueError("field_batch_size must be a positive integer.")
        batch_size = 1 if mode == "cached_serial" else (field_batch_size or count)
        started = time.perf_counter()
        mx.reset_peak_memory()
        memory_before = mx.get_active_memory()
        rows = []
        prefill_seconds = 0.0
        prefix_cache = None
        if mode != "independent":
            prefill_started = time.perf_counter()
            prefix_cache = self.model.make_cache()
            h = self.backbone(mx.array([prepared.prefix_ids]), cache=prefix_cache)
            mx.eval(h[:, -1], [c.state for c in prefix_cache])
            del h
            prefill_seconds = time.perf_counter() - prefill_started
        branches_started = time.perf_counter()
        if mode == "independent":
            for ids, candidates in zip(prepared.full_ids, prepared.candidate_ids):
                cache = self.model.make_cache()
                h = self.backbone(mx.array([ids]), cache=cache)[:, -1, :]
                logits = candidate_projection(h, self.head_weight, candidates)[0]
                mx.eval(logits)
                rows.append(logits)
                del h, cache
            batches = count
        else:
            batches = 0
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            for start in range(0, count, batch_size):
                tails = prepared.suffix_ids[start:start + batch_size]
                lengths = [len(t) for t in tails]
                width = max(lengths)
                tokens = mx.array([t + [pad_id] * (width - len(t)) for t in tails])
                cache = broadcast_cache(prefix_cache, len(tails))
                # Right padding is after the scored position. Causality prevents it
                # affecting valid outputs; these branch caches are then discarded.
                h = self.backbone(tokens, cache=cache)
                last = h[mx.arange(len(tails)), mx.array(lengths) - 1, :]
                candidate_sets = prepared.candidate_ids[start:start + len(tails)]
                union = sorted(set(token for ids in candidate_sets for token in ids))
                projected = candidate_projection(last, self.head_weight, union)
                mx.eval(projected)
                lookup = {token: i for i, token in enumerate(union)}
                rows.extend(projected[i, mx.array([lookup[t] for t in ids])]
                            for i, ids in enumerate(candidate_sets))
                mx.eval(rows)
                del h, last, projected, cache
                batches += 1
        branch_seconds = time.perf_counter() - branches_started
        for row in rows:
            if not mx.all(mx.isfinite(row)).item():
                raise ValueError("Model produced non-finite candidate logits.")
        stats = {
            "mode": mode, "fields": count, "prefix_tokens": len(prepared.prefix_ids),
            "suffix_tokens": list(map(len, prepared.suffix_ids)),
            "prefix_prefill_count": count if mode == "independent" else 1,
            "branch_batches": batches,
            "prefill_seconds": prefill_seconds,
            "field_evaluation_seconds": branch_seconds,
            "total_seconds": time.perf_counter() - started,
            "mlx_peak_active_gib": mx.get_peak_memory() / 2**30,
            "mlx_active_before_gib": memory_before / 2**30,
            "mlx_peak_above_start_gib": max(0, mx.get_peak_memory() - memory_before) / 2**30,
            "full_vocabulary_projection": False,
        }
        return rows, stats

    def score(self, context, schema, mode=None, field_batch_size=None):
        mode = mode or getattr(self, "default_mode", "parallel")
        prepared = self.prepare(context, schema)
        rows, stats = self.evaluate(prepared, mode=mode, field_batch_size=field_batch_size)
        fields, output = {}, {}
        for name, choices, ids, row in zip(prepared.names, prepared.choices, prepared.candidate_ids, rows):
            probs = mx.softmax(row / self.temperature)
            mx.eval(probs)
            best = mx.argmax(row).item()
            keys = [choice_key(v) for v in choices]
            output[name] = choices[best]
            fields[name] = {
                "value": choices[best], "scores": dict(zip(keys, probs.tolist())),
                "logits": dict(zip(keys, row.tolist())),
                "candidate_token_ids": ids,
                "code_to_choice": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices)),
            }
        model_id, revision = getattr(self, "model_id", MODEL_ID), getattr(self, "revision", REVISION)
        return {"model": model_id, "revision": revision, "backend": "mlx",
                "temperature": self.temperature,
                "temperature_fitted": self.temperature == fitted_temperature(model_id, revision),
                "context": context, "output": output, "fields": fields, "metrics": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--schema")
    group.add_argument("--schema-file", type=Path)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--field-batch-size", type=int)
    parser.add_argument("--mode", choices=["parallel", "cached_serial", "independent"], default="parallel")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        schema = parse_schema(args.schema if args.schema is not None else args.schema_file.read_text())
        if not args.context.strip():
            raise ValueError("Context must be nonempty.")
        scorer = ParallelScorer(max_input_tokens=args.max_input_tokens, temperature=args.temperature)
        result = scorer.score(args.context, schema, args.mode, args.field_batch_size)
        import json
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
