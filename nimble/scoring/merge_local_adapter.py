"""Merge a local adapter with its pinned, cached base for Mac/Linux inference."""
import argparse
import hashlib
import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path

from nimble.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path,
                        help='Local snapshot of the pinned base; defaults to the Hugging Face hub cache layout')
    args = parser.parse_args()
    contract = json.loads((args.adapter/'schema_config.json').read_text())
    prompt_hash = hashlib.sha256((PROJECT_ROOT/'nimble/scoring/parallel_schema.py').read_bytes()).hexdigest()
    assert contract['prompt_code_sha256'] == prompt_hash
    base_path = args.base or PROJECT_ROOT/'.cache/huggingface/hub'/('models--'+contract['model'].replace('/', '--'))/'snapshots'/contract['revision']
    if not base_path.is_dir():
        raise FileNotFoundError(f'Pinned base must already be cached: {base_path}')
    if args.base is not None:
        if not (args.base/'config.json').is_file():
            raise FileNotFoundError(f'--base needs a complete snapshot with config.json: {args.base}')
        named = json.loads((args.base/'config.json').read_text(encoding='utf-8')).get('_name_or_path', '')
        if '/' in named and named != contract['model']:
            raise ValueError(f'--base config names {named!r}, but the adapter contract pins {contract["model"]!r}')
    base_path = base_path.resolve()
    adapter_hash = hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
    ready = args.output/'READY.json'
    if ready.exists():
        saved = json.loads(ready.read_text())
        assert (saved['adapter_sha256'] == adapter_hash and saved['base_revision'] == contract['revision']
                and saved.get('base_path') == str(base_path))
        print('Reusing verified merge:', args.output)
        return
    if args.output.exists():
        raise FileExistsError('Output exists without READY.json; use a fresh directory')
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(8)
    started = time.perf_counter()
    from nimble.training.candidate_schema import validate_contract
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    validate_contract(contract, tokenizer)
    print('Loading cached base on CPU', flush=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(base_path, local_files_only=True,
                dtype=torch.bfloat16, device_map='cpu', attn_implementation='sdpa', trust_remote_code=False)
    print('Merging adapter in BF16', flush=True)
    merged = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).merge_and_unload(safe_merge=True)
    print('Saving merged checkpoint', flush=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(args.output)
    shutil.copyfile(args.adapter/'schema_config.json', args.output/'schema_config.json')
    hashes = {}
    for path in sorted(args.output.glob('*.safetensors')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    manifest = {'base_model': contract['model'], 'base_revision': contract['revision'], 'base_path': str(base_path),
                'adapter_sha256': adapter_hash, 'prompt_sha256': prompt_hash,
                'training_examples': contract['data_audit']['training_rows'],
                'heldout_examples': contract['data_audit']['validation_rows'],
                'precision': 'bfloat16', 'quantized': False, 'merge': 'PEFT safe_merge on CPU',
                'merge_seconds': time.perf_counter()-started, 'weight_sha256': hashes,
                'versions': {p: version(p) for p in ('torch', 'peft', 'transformers')}}
    ready.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'weight_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
