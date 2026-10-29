"""Frozen paired analysis and independent replay of final reported layouts."""
import gzip
import hashlib
import itertools
import json
import platform
import importlib.metadata
from collections import defaultdict
from pathlib import Path
import numpy as np
from pymoo.indicators.hv import HV
import water_plant_layout_v3 as model
from revision_decoders import decode
from revision_evaluation import evaluate_layout, EvaluationConfig
from revision_pymoo_baselines import vector_to_layout

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'paper/revision_final_experiments'
SCALES = np.array([15000,1000,10000,1])


def sign_flip(differences):
    d = np.asarray(differences,float)
    observed = abs(d.mean())
    values = [abs(np.mean(d*np.array(signs))) for signs in itertools.product([-1,1],repeat=len(d))]
    return sum(v >= observed-1e-12 for v in values)/len(values)


def main():
    manifest = json.loads((OUT/'manifest.json').read_text())
    digest = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    assert sign_flip([1]*10) == 2/1024
    assert sign_flip([0]*10) == 1
    assert HV(ref_point=np.array([3,3]))(np.array([[1,2],[2,1]])) == 3
    files = sorted(OUT.glob('*.json.gz'))
    assert len(files) == len(manifest['jobs']) == 60, 'wait for all frozen jobs'
    groups = defaultdict(list)
    baseline_points = []
    initial_baselines,initial_actions = {},{}
    replay_count = 0
    fs = model.create_facilities()
    for path in files:
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            saved = json.load(stream)
        assert saved['manifest_sha256'] == digest
        s,r = saved['summary'],saved['result']
        config = s['settings']
        assert r['evaluation_count'] == 1536
        if s['family'] == 'baseline':
            key = config['algorithm']+'_'+config['decoder']
            best = r['best_by_reported_weights']
            if config['seed'] in initial_baselines:
                assert r['initial_vectors'] == initial_baselines[config['seed']]
            initial_baselines[config['seed']] = r['initial_vectors']
            layout,_,_ = vector_to_layout(best['vector'],fs,config['decoder'],config['coordinate_scale_factor'])
            points = [a['evaluation']['objectives'] for a in r['archive']]
            assert np.all(np.isfinite(points)) and np.all(np.asarray(points) >= 0)
            for i,p in enumerate(points):
                assert not any(all(x<=y for x,y in zip(q,p)) and any(x<y for x,y in zip(q,p)) for j,q in enumerate(points) if i!=j)
            s['points'] = points
            baseline_points.extend(points)
            feasible = [e for e in r['evaluations'] if e['evaluation']['feasible']]
            s['feasible_fraction'] = len(feasible)/1536
            scores = [e['score'] for e in r['evaluations'][:768] if e['score'] is not None]
        else:
            key = 'action_'+config['controller']
            best = r['best']
            if config['seed'] in initial_actions:
                assert r['initial_chromosomes'] == initial_actions[config['seed']]
            initial_actions[config['seed']] = r['initial_chromosomes']
            layout,_,_ = decode(fs,best['chromosome'],None,model.get_safety_distance)
            s['feasible_fraction'] = sum(e['feasible'] for e in r['evaluations'])/1536
            scores = [e['score'] for e in r['evaluations'][:768] if e['score'] is not None]
            assert len(r['transitions']) == (1536-16)//2
            assert all(t['child_evaluation'] == t['base_evaluation']+1 and t['terminal'] for t in r['transitions'])
        check = evaluate_layout(layout,fs,EvaluationConfig(None))
        assert check['metrics'] == best['evaluation']['metrics']
        assert check['geometric_feasible']
        replay_count += 1
        s['last_half_relative_gain'] = (min(scores)-s['score'])/min(scores)
        groups[key].append(s)
    maximum = np.max(np.asarray(baseline_points)/SCALES,axis=0)
    references = [np.maximum(1,maximum*factor) for factor in [1.1,1.2]]
    for rows in groups.values():
        for row in rows:
            if 'points' in row:
                points = np.asarray(row['points'])/SCALES
                row['hypervolume'],row['hypervolume_sensitivity'] = [float(HV(ref_point=ref)(points)) for ref in references]
    comparisons = []
    for label,treatment,control,metric in [
        ('GA decoder','ga_rrp','ga_coordinate_repair','score'),
        ('NSGA-II decoder','nsga2_rrp','nsga2_coordinate_repair','hypervolume'),
        ('Action learning','action_q_learning','action_random','score')]:
        a = sorted(groups[treatment],key=lambda r:r['settings']['seed'])
        b = sorted(groups[control],key=lambda r:r['settings']['seed'])
        differences = [x[metric]-y[metric] for x,y in zip(a,b)]
        lower_better = metric == 'score'
        comparisons.append({'label':label,'treatment':treatment,'control':control,'metric':metric,
            'mean_difference':float(np.mean(differences)),
            'mean_paired_relative_improvement':float(np.mean([(y[metric]-x[metric])/y[metric] if lower_better else (x[metric]-y[metric])/y[metric] for x,y in zip(a,b)])),
            'wins':sum(d < 0 if lower_better else d > 0 for d in differences),
            'p_raw':sign_flip(differences),'differences':differences})
    running = 0
    for rank,c in enumerate(sorted(comparisons,key=lambda c:c['p_raw'])):
        running = max(running,min(1,(3-rank)*c['p_raw']))
        c['p_holm'] = running
    statistics = {}
    for key,rows in groups.items():
        assert len(rows) == 10
        statistics[key] = {}
        for metric in ['score','feasible_fraction','hypervolume','hypervolume_sensitivity','last_half_relative_gain']:
            if metric in rows[0]:
                data = [r[metric] for r in rows]
                statistics[key][metric] = {'mean':float(np.mean(data)),'sd':float(np.std(data,ddof=1)),'min':float(min(data)),'max':float(max(data))}
        for metric in rows[0]['metrics']:
            data = [r['metrics'][metric] for r in rows]
            statistics[key][metric] = {'mean':float(np.mean(data)),'sd':float(np.std(data,ddof=1)),'min':float(min(data)),'max':float(max(data))}
    analysis = {'run_count':len(files),'total_evaluations':60*1536,'replayed_layouts':replay_count,
        'statistics':statistics,'comparisons':comparisons,'hv_reference':references[0].tolist(),
        'hv_reference_sensitivity':references[1].tolist(),'groups':groups,
        'test_assumption':'paired sign-flip inference assumes exchangeability of within-seed signs under the null; seed replication pertains only to this fixed instance and protocol',
        'plateau_count':sum(r['last_half_relative_gain'] < .01 for rows in groups.values() for r in rows)}
    analysis['environment'] = {'python':platform.python_version(),'platform':platform.platform(),
        'processor':platform.processor(),'packages':{name:importlib.metadata.version(name) for name in ['numpy','pymoo','scipy']}}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    print(json.dumps({k:analysis[k] for k in ['run_count','replayed_layouts','comparisons','hv_reference','plateau_count']},indent=2))


if __name__ == '__main__':
    main()
