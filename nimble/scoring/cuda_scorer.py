"""CUDA and ROCm counterpart of the MLX independent candidate scorer.

Shares prompt construction and projects only candidate output rows in FP32.
Checkpoint weights stay in the load dtype, BF16 by default; no generated text or
probability fitting. An accelerate device map can stream weights from host
memory when the checkpoint does not fit on the GPU.
"""

import hashlib
import json
import math
import time
from collections import Counter

import torch

from nimble.scoring.calibration import fitted_temperature
from nimble.scoring.parallel_schema import choice_key, prepare_prompts

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
HOST_MEMORY_BUDGET = "512GiB"


def physical_weight(module):
    """Return a module's weight even when accelerate offloaded it to host memory.

    Offloaded parameters are replaced by meta tensors; the values live in the hook's
    weights map, which the forward hooks stream to the GPU on demand. Reading the
    head directly for the candidate projection bypasses those hooks.
    """
    weight = module.weight
    if weight.device.type != "meta":
        return weight
    weights = getattr(getattr(module, "_hf_hook", None), "weights_map", None)
    try:
        return weights["weight"]
    except (TypeError, KeyError):
        raise RuntimeError("Output head is on the meta device without an offload weights map") from None


def execution_device(module):
    """Where a module runs: its hook's device under a device map, else its weight's device."""
    device = getattr(getattr(module, "_hf_hook", None), "execution_device", None)
    return torch.device(device) if device is not None else module.weight.device


def candidate_projection(hidden, weight, token_ids):
    """Project only the candidate rows, in FP32, on the hidden state's device."""
    indices = torch.tensor(token_ids, device=weight.device)
    rows = weight.index_select(0, indices).to(hidden.device)
    return hidden.float() @ rows.float().T


class CudaCandidateScorer:
    def __init__(self, model_path, model_id, revision, max_input_tokens=4096, temperature=None,
                 device_map=None, max_gpu_memory=None, dtype="bfloat16", attention=None):
        if not torch.cuda.is_available():
            raise RuntimeError("This runner requires a CUDA or ROCm GPU")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(DTYPES)}")
        dtype = DTYPES[dtype]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU reports no BF16 support; pass dtype=float16 to override")
        if not isinstance(max_input_tokens, int) or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        if temperature is None:
            # The checkpoint's fitted temperature when it has one, else raw probabilities.
            temperature = fitted_temperature(model_id, revision) or 1.0
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if attention not in (None, "sdpa", "eager"):
            raise ValueError("attention must be sdpa or eager")
        if max_gpu_memory is not None and device_map is None:
            raise ValueError("max_gpu_memory requires a device_map")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

        # Avoid TF32 rounding in the small FP32 candidate projection.
        torch.backends.cuda.matmul.allow_tf32 = False
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if config.model_type not in {"qwen3_5", "gemma3_text"}:
            raise ValueError(f"Unsupported architecture: {config.model_type}")
        cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else AutoModelForCausalLM
        # Torch 2.8's optimized CUDA SDPA produced incorrect Gemma outputs once
        # prompts crossed its 512-token sliding window. Eager agrees with the
        # CPU and math-only SDPA checks; retain that validated path for Gemma.
        attention = attention or ("eager" if config.model_type == "gemma3_text" else "sdpa")
        loading_options = dict(local_files_only=True, dtype=dtype, attn_implementation=attention,
                               output_loading_info=True)
        if device_map is not None:
            loading_options["device_map"] = device_map
            if max_gpu_memory is not None:
                # Weights beyond the GPU budget stay in host memory and stream through hooks.
                loading_options["max_memory"] = {index: max_gpu_memory for index in range(torch.cuda.device_count())}
                loading_options["max_memory"]["cpu"] = HOST_MEMORY_BUDGET
        self.model, loading = cls.from_pretrained(model_path, **loading_options)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint did not load exactly: {loading}")
        self.model = self.model.eval()
        if device_map is None:
            self.model = self.model.to("cuda")
        elif not any(p.device.type == "cuda" for p in self.model.parameters() if p.device.type != "meta"):
            raise ValueError("device_map placed no weights on a CUDA device")
        self.backbone = self.model.model
        self.head_weight = physical_weight(self.model.get_output_embeddings())
        self.device = execution_device(self.model.get_input_embeddings())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.system_role = config.model_type != "gemma3_text"
        self.model_id, self.revision = model_id, revision
        text_config = getattr(config, "text_config", config)
        self.max_input_tokens = min(max_input_tokens, text_config.max_position_embeddings - 1)
        self.temperature = temperature
        placement = Counter(str(d) for d in getattr(self.model, "hf_device_map", {}).values())
        properties = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
        self.runtime = {
            "dtype": str(dtype).removeprefix("torch."), "attention": attention,
            "device_map": device_map if device_map is None or isinstance(device_map, str) else "custom",
            "max_gpu_memory": max_gpu_memory, "placement": dict(placement) or None,
            "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
            "gpus": [p.name for p in properties],
        }

    def prepare(self, context, schema):
        return prepare_prompts(self.tokenizer, context, schema, self.max_input_tokens,
                               system_role=self.system_role)

    @torch.inference_mode()
    def score(self, context, schema, mode="independent"):
        if mode != "independent":
            raise ValueError("CUDA runner currently supports independent full-prompt prefill only")
        prepared = self.prepare(context, schema)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        fields, output = {}, {}
        for name, choices, ids, candidates in zip(prepared.names, prepared.choices,
                                                   prepared.full_ids, prepared.candidate_ids):
            tokens = torch.tensor([ids], device=self.device)
            hidden = self.backbone(input_ids=tokens, use_cache=False).last_hidden_state[:, -1, :]
            logits = candidate_projection(hidden, self.head_weight, candidates)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite candidate logits")
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
            best = logits.argmax().item()
            keys = [choice_key(value) for value in choices]
            output[name] = choices[best]
            fields[name] = {
                "value": choices[best], "scores": dict(zip(keys, probabilities.tolist())),
                "logits": dict(zip(keys, logits.tolist())), "candidate_token_ids": candidates,
                "code_to_choice": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices)),
                "prompt_token_count": len(ids),
                "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
            del hidden, logits, probabilities
        torch.cuda.synchronize()
        return {"model": self.model_id, "revision": self.revision, "backend": "cuda",
                "temperature": self.temperature,
                "temperature_fitted": self.temperature == fitted_temperature(self.model_id, self.revision),
                "runtime": self.runtime,
                "context": context, "output": output, "fields": fields,
                "metrics": {"mode": mode, "fields": len(fields),
                            "total_seconds": time.perf_counter() - started,
                            "cuda_peak_active_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "full_vocabulary_projection": False}}
