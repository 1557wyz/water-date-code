"""Frozen paired experiments for the manuscript revision, separate algorithm families."""
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import water_plant_layout_v3 as model
from revision_evaluation import EvaluationConfig
from revision_pymoo_baselines import BaselineConfig, run_baseline
from revision_ga import GAConfig, run_ga

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'paper/revision_final_experiments'
SEEDS = list(range(1101,1111))


def run_one(job):
    family, settings, digest = job
    name = f"{family}_{settings.get('algorithm',settings.get('controller'))}_{settings['decoder']}_{settings['seed']}"
    path = OUT/(name+'.json.gz')
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            saved = json.load(stream)
        assert saved['manifest_sha256'] == digest
        return saved['summary']
    if family == 'baseline':
        config = BaselineConfig(**settings)
        def progress(state):
            (OUT/(name+'.progress.json')).write_text(json.dumps(state),encoding='utf-8')
        result = run_baseline(model.create_facilities(),EvaluationConfig(None),config,progress)
        assert result['evaluation_count'] == result['library_evaluation_count'] == config.evaluation_budget
        best = result['best_by_reported_weights']
        metrics = best['evaluation']['metrics']
        for r in result['evaluations']:
            r['evaluation'] = {k:r['evaluation'][k] for k in ['feasible','objectives','metrics','errors']}
    else:
        config = GAConfig(**settings)
        result = run_ga(model.create_facilities(),EvaluationConfig(None),config)
        assert result['evaluation_count'] == config.evaluation_budget
        best = result['best']
        metrics = best['evaluation']['metrics']
    summary = {'family':family,'settings':settings,'file':path.name,'metrics':metrics,
               'score':best['score'],'runtime_seconds':result['runtime_seconds'],
               'evaluations':result['evaluation_count'],'archive_size':len(result['archive'])}
    saved = {'manifest_sha256':digest,'summary':summary,'result':result}
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as stream:
        json.dump(saved,stream,allow_nan=False)
    tmp.replace(path)
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    jobs = []
    for seed in SEEDS:
        for algorithm in ['ga','nsga2']:
            for decoder in ['coordinate_repair','rrp']:
                jobs.append(('baseline',asdict(BaselineConfig(algorithm,decoder,16,1536,seed,
                             (1,1,1,1),(15000,1000,10000,1),.75 if decoder=='coordinate_repair' else 1,'random'))))
        for controller in ['random','q_learning']:
            jobs.append(('action',asdict(GAConfig(16,1536,seed,'rrp',controller,(1,1,1,1),(15000,1000,10000,1)))))
    manifest = {'seeds':SEEDS,'jobs':jobs,'workers':4,'facility_count':34,'boundary':None,
                'scope':'single-instance fixed-budget paired comparison; two separate algorithm families',
                'analysis_protocol':{
                    'primary_comparisons':['GA RRP vs GA coordinate repair scalar score','NSGA2 RRP vs NSGA2 coordinate repair hypervolume','RRP action learning vs random action scalar score'],
                    'test':'two-sided exact paired sign-flip test of mean differences; Holm correction over three primary tests',
                    'hypervolume':'four objectives divided by fixed scales; common reference 1.1 times coordinatewise maximum of union of all baseline archives, with each reference component at least 1; sensitivity factor 1.2',
                    'representative':'minimum equal-weight normalized sum; post-hoc only for NSGA2',
                    'limits':'10 paired seeds do not establish population-wide generalization, convergence, or equal search-space representation; no power claim'},
                'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__)]+[Path(__file__).with_name(n) for n in ['revision_ga.py','revision_rl.py','revision_pymoo_baselines.py','revision_coordinate_repair.py','revision_decoders.py','revision_evaluation.py','water_plant_layout_v3.py']]}}
    raw = json.dumps(manifest,sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    path = OUT/'manifest.json'
    if path.exists():
        assert json.dumps(json.loads(path.read_text()),sort_keys=True) == raw
    else:
        path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summaries = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(run_one,(family,settings,digest)) for family,settings in jobs]
        for future in as_completed(pending):
            summaries.append(future.result())
            (OUT/'summary.json').write_text(json.dumps({'completed':len(summaries),'planned':len(jobs),'runs':summaries},indent=2),encoding='utf-8')
            print(f"{len(summaries)}/{len(jobs)} completed {summaries[-1]['file']}",flush=True)


if __name__ == '__main__':
    main()
