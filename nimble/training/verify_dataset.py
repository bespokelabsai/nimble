"""Check the canonical training release without a GPU, tokenizer, or API calls."""
import argparse
import hashlib
import json
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.training.evidence_data import validate_evidence_data
from nimble.training.schema_data import fingerprint, read_rows, validate_separation


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('storage_format') != 'nimble-self-contained-v1':
        raise ValueError('Expected the canonical self-contained release')
    rows = {}
    for name, count in [('train.jsonl', 2676), ('eval.jsonl', 324)]:
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['file_sha256'][name]:
            raise ValueError('Frozen dataset bytes changed: ' + name)
        rows[name] = read_rows(path)
        if len(rows[name]) != count:
            raise ValueError('Unexpected dataset size: ' + name)
    training, evaluation = rows['train.jsonl'], rows['eval.jsonl']
    if manifest['training_rows'] != len(training) or manifest['eval_rows'] != len(evaluation):
        raise ValueError('Manifest dataset counts differ from the files')
    if fingerprint(training) != manifest['train_sha256']:
        raise ValueError('Training fingerprint changed')
    if fingerprint(evaluation) != manifest['eval_record_fingerprint']:
        raise ValueError('Evaluation fingerprint changed')
    if {r['id'] for r in training} & {r['id'] for r in evaluation}:
        raise ValueError('Training/evaluation IDs overlap')
    validate_separation(training, evaluation)
    validate_evidence_data(training, manifest)
    return {'training_rows': len(training), 'holdout_rows': len(evaluation),
            'sha256': manifest['file_sha256'], 'certificate_checks': 'passed',
            'source_family_overlap': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data')
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir), indent=2))


if __name__ == '__main__':
    main()
