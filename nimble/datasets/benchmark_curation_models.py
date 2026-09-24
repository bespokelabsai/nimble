"""Time identical accepted-training-example targets on explicit model APIs."""
import asyncio
import copy
import json
import os
import time
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import configure, read_jsonl
from nimble.datasets.contrastive_data import fingerprint
from nimble.datasets.scale_evidence_dataset import atomic_json
from nimble.datasets.scaled_evidence import quotas
from nimble.datasets.fast_training_dataset import parser, load_training_plans, AsyncStages, curate, curation_executor, resume_policy
from nimble.datasets.curation_providers import CurationClient, provider_for


def configuration(args, sources, lineage, model):
    files = [Path(__file__), Path(__file__).with_name('curation_providers.py'),
             Path(__file__).with_name('curation_serialization.py')]
    files += [Path(__file__).with_name(n) for n in ('fast_training_dataset.py','curation_profiles.py',
        'scaled_evidence.py','scale_evidence_dataset.py','scaled_evidence_stages.py',
        'evidence_stages.py','curate_paired_evidence.py','evidence_curation.py','contrastive_data.py')]
    config = {'runner_version': 'model-benchmark-v2', 'source': str(args.source.resolve()),
        'source_sha256': fingerprint(sources), 'rule_plan_lineage': lineage,
        'generator_model': model, 'verifier_model': model, 'provider': provider_for(model, args.claude_provider),
        'draft_reasoning': args.draft_reasoning, 'check_reasoning': args.check_reasoning,
        'target': args.target, 'seed': args.seed, 'max_edit_words': args.max_edit_words,
        'attempt_multiplier': args.attempt_multiplier,
        'implementation_sha256': fingerprint({p.name:p.read_text() for p in files})}
    if args.guidance_from_variation:
        config['guidance_from_variation'] = args.guidance_from_variation
    if args.resume_policy:
        config['resume_policy'] = resume_policy(args)
    return config


async def benchmark_one(args, sources, plans, config):
    config_path = args.output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing run settings differ; use another output directory')
    if not args.offline:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)
    client = None if args.offline else CurationClient([args.generator_model], claude_provider=args.claude_provider)
    stage = AsyncStages(args.output, args, client)
    started = time.monotonic()
    try:
        manifest = await curate(args, sources, plans, config, stage)
        performance = {'model': args.generator_model, 'provider': config['provider'],
            'status': 'complete', 'elapsed_seconds': time.monotonic()-started,
            'accepted_examples': manifest['training']['examples'], 'attempted_groups': manifest['attempted_groups'],
            'api_calls_this_run': stage.api_calls, 'cache_hits_this_run': stage.cache_hits,
            'usage_this_run': dict(stage.usage), 'draft_reasoning': args.draft_reasoning,
            'check_reasoning': args.check_reasoning, 'source': str(args.output.resolve())}
        if not args.offline:
            atomic_json(args.output/'performance.json', performance)
        else:
            # Never replace the measured online timing with a cached replay time.
            performance = json.loads((args.output/'performance.json').read_text())
        return performance
    except Exception as error:
        performance = {'model': args.generator_model, 'provider': config['provider'],
            'status': 'failed', 'elapsed_seconds': time.monotonic()-started,
            'api_calls_this_run': stage.api_calls, 'cache_hits_this_run': stage.cache_hits,
            'usage_this_run': dict(stage.usage), 'error': str(error)}
        if not args.offline:
            atomic_json(args.output/'failed_run.json', performance)
        raise
    finally:
        if client:
            await client.close()


def write_report(output, results):
    atomic_json(output/'benchmark.json', {'results': results, 'notes': [
        'One accepted base/counterfactual pair per model by default; not a broad quality or throughput evaluation.',
        'Timing starts after loading reused Sol training-rule plans and client setup; includes rejected candidates and verification.',
        'Model serves as both generator and verifier in separate calls. Same prompts/gates/seed/source pool.',
        'Low/medium effort settings are provider-specific and do not imply equal reasoning compute.',
        'Models run sequentially. Provider retries/schema compilation are included; token export/loader audit are excluded.',
        'Nonzero application cache hits on resume must be considered before treating timing as a fresh-run measurement.']})
    lines=['# Training curation timing', '', '| Model | Accepted rows | Seconds | Attempted pairs | API calls | Cache hits |',
           '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['model']} | {r.get('accepted_examples', 'failed')} | {r['elapsed_seconds']:.2f} | "
                     f"{r.get('attempted_groups', '—')} | {r['api_calls_this_run']} | {r['cache_hits_this_run']} |")
    lines += ['', 'These are individual runs of the same small target, not average performance or independent label-quality measurements.',
              'Each model generates and verifies its own cases. Audited rules are reused from training-only Sol data.',
              'See `benchmark.json` for timing boundaries and each model directory for data and verification certificates.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=parser()
    p.description=__doc__
    p.set_defaults(target=2, attempt_multiplier=32, output=PROJECT_ROOT/'data/curation_model_benchmark')
    p.add_argument('--models', nargs='+', default=['gpt-5.6-luna','claude-sonnet-5'])
    p.add_argument('--claude-provider', choices=['anthropic','openrouter','requesty'], default='anthropic')
    args=p.parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError('Models must be unique')
    if min(args.max_edit_words,args.attempt_multiplier,args.concurrency,args.group_concurrency,
           args.requests_per_minute,args.tokens_per_minute) <= 0:
        raise ValueError('All limits must be positive')
    sources=read_jsonl(args.source)
    quotas(sources,args.target)
    plans,lineage=load_training_plans(args.plans,sources)
    configs={m:configuration(args,sources,lineage,m) for m in args.models}
    if args.dry_run:
        print(json.dumps(configs,indent=2));return
    if not args.offline:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT/'.env',override=False)
        configure()
        for provider in {provider_for(m,args.claude_provider) for m in args.models}:
            key={'openai':'OPENAI_API_KEY','anthropic':'ANTHROPIC_API_KEY','openrouter':'OPENROUTER_API_KEY','requesty':'REQUESTY_API_KEY'}[provider]
            if not os.environ.get(key):raise ValueError(key+' is required')
        args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for model in args.models:
        run_args=copy.copy(args)
        run_args.generator_model=run_args.verifier_model=model
        run_args.output=args.output/model
        print(f'Starting {model}: target {args.target} accepted examples.',flush=True)
        try:
            with asyncio.Runner() as runner:
                runner.get_loop().set_default_executor(curation_executor(args.group_concurrency))
                result=runner.run(benchmark_one(run_args,sources,plans,configs[model]))
            results.append(result)
            print(json.dumps(result),flush=True)
        except Exception:
            if (run_args.output/'failed_run.json').exists():
                results.append(json.loads((run_args.output/'failed_run.json').read_text()))
            else:
                raise
        if not args.offline:
            write_report(args.output,results)
    if any(r['status']!='complete' for r in results):
        raise SystemExit('At least one model did not complete; see failed_run.json.')


if __name__=='__main__':main()
