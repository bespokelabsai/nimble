"""Pinned Nimble checkpoint, CPU merge, and public SGLang service."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "bespokelabs/Bespoke-Nimble-9B"
REVISION = "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"
UPSTREAM = "7f84bedc169439f03379c2fa8d00ada220af2295"
MERGED = f"/models/openjeff-{REVISION}"
app = modal.App("nimble-sglang")
volume = modal.Volume.from_name("openjeff-models", create_if_missing=True)

merge_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
    "huggingface-hub==1.32.0",
).env({"HF_HOME": "/models/huggingface"})


@app.function(image=merge_image, cpu=8, memory=98304, timeout=3600,
              volumes={"/models": volume}, secrets=[modal.Secret.from_name("openjeff-huggingface")])
def prepare_model():
    import hashlib
    import json
    import os
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    output = Path(MERGED)
    if (output / "READY.json").exists():
        return str(output)
    adapter = Path(snapshot_download(MODEL, revision=REVISION))
    contract = json.loads((adapter / "schema_config.json").read_text())
    prompt_hash = hashlib.sha256((adapter / "parallel_schema.py").read_bytes()).hexdigest()
    if prompt_hash != contract["prompt_code_sha256"]:
        raise ValueError("Published prompt does not match the training contract")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        contract["model"], revision=contract["revision"], dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="sdpa", trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    temporary = Path(str(output) + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(adapter).save_pretrained(temporary)
    shutil.copy(adapter / "schema_config.json", temporary)
    (temporary / "READY.json").write_text(json.dumps({
        "model": MODEL, "revision": REVISION, "base": contract["model"],
        "base_revision": contract["revision"], "prompt_sha256": prompt_hash,
        "precision": "bfloat16", "merge": "PEFT safe_merge",
    }, indent=2))
    os.rename(temporary, output)
    volume.commit()
    return str(output)


image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.19-cu130").entrypoint([])
    .uv_pip_install("uv==0.12.5")
    .run_commands(
        "uv venv --python 3.12 /opt/nimble-api",
        f"uv pip install --python /opt/nimble-api/bin/python "
        f"'openjev-sglang @ https://github.com/ekzhang/openjev-sglang/archive/{UPSTREAM}.tar.gz' "
        "'transformers==5.17.0' 'huggingface-hub==1.32.0'",
    )
    .add_local_file(ROOT / "nimble/__init__.py", "/opt/app/nimble/__init__.py", copy=True)
    .add_local_file(ROOT / "nimble/compat.py", "/opt/app/nimble/compat.py", copy=True)
    .add_local_file(ROOT / "nimble/scoring/__init__.py", "/opt/app/nimble/scoring/__init__.py", copy=True)
    .add_local_file(ROOT / "nimble/scoring/calibration.py", "/opt/app/nimble/scoring/calibration.py", copy=True)
    .add_local_file(ROOT / "nimble/scoring/parallel_schema.py", "/opt/app/nimble/scoring/parallel_schema.py", copy=True)
    .add_local_dir(ROOT / "nimble/serving", "/opt/app/nimble/serving", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": "/opt/app", "HF_HOME": "/models/huggingface",
          "NIMBLE_MODEL_PATH": MERGED, "TOKENIZERS_PARALLELISM": "false",
          "SGLANG_CACHE_DIR": "/models/runtime/sglang", "TRITON_CACHE_DIR": "/models/runtime/triton"})
)


# The exclamation mark prevents Modal from substituting an H200 during benchmarks.
@app.server(image=image, gpu="H100!", cpu=8, memory=49152,
            volumes={"/models": volume}, port=8000, routing_region="us-west",
            unauthenticated=True, min_containers=0, max_containers=1,
            scaledown_window=120, startup_timeout=1200, exit_grace_period=30)
class Nimble:
    @modal.enter()
    def startup(self):
        import subprocess
        print("Serving GPU: " + subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(), flush=True)
        if not Path(MERGED, "READY.json").exists():
            raise RuntimeError("Run prepare_model before deploying the server")
        self.process = subprocess.Popen(
            ["/opt/nimble-api/bin/python", "-m", "nimble.serving.server"], start_new_session=True
        )
        # Fail the container if either the API or backend exits unexpectedly.
        import threading
        import os
        def watch():
            self.process.wait()
            os._exit(1)
        threading.Thread(target=watch, daemon=True).start()

    @modal.exit()
    def shutdown(self):
        import os
        import signal
        if hasattr(self, "process"):
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


@app.local_entrypoint()
def main():
    print(prepare_model.remote())
