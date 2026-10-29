"""Frozen pure graph-process/RRP + original native GA validation; resumable."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONHASHSEED']='0'
import argparse
import contextlib
import copy
import gzip
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.dont_write_bytecode=True
from common import ROOT,CORE,digest,load,save
from graph_model import make_optimizer,rename_instance
from validation import audit,serialize
from process_model import evaluate
from process_initializer import construct

PROTOCOL=ROOT/'config/frozen_pure_protocol.json'
ENGINES=('common.py','graph_model.py','process_decoder.py','rrp_decoder.py','process_model.py','validation.py','run_pure.py',
        'process_initializer.py','geometric_repair.py','evolve_core.py')

def controlled(ins,parallel):
    d=copy.deepcopy(ins);d['id']=f'PARALLEL{parallel}';d['group']='controlled_parallel';d['data_status']='controlled benchmark expansion, excluded from cross-plant case count'
    group=('MB','FB','SB','FT');table={f['id']:f for f in d['facilities']}
    for i in range(2,parallel+1):
        for name in group:
            f=copy.deepcopy(table[name]);f['id']=f'{name}_p{i}';f['label']+=f' parallel {i}';d['facilities'].append(f)
            d['elevations'][f['id']]=copy.deepcopy(d['elevations'][name])
    original={(e['source'],e['target']):e for e in ins['process_edges']}
    heads={(e['source'],e['target']):e for e in ins['hydraulic_segments']}
    affected=(('PS','MB'),('MB','FB'),('FB','SB'),('SB','FT'),('FT','CWT'),('FT','OZG'),('SB','STK'))
    for i in range(2,parallel+1):
        for a,b in affected:
            source=f'{a}_p{i}' if a in group else a;target=f'{b}_p{i}' if b in group else b
            e=copy.deepcopy(original[a,b]);e.update(source=source,target=target);d['process_edges'].append(e)
            h=copy.deepcopy(heads[a,b]);h.update(source=source,target=target);d['hydraulic_segments'].append(h)
    d['contacts']=copy.deepcopy(d['process_edges']);d['facility_count']=len(d['facilities'])
    return d

def prepare():
    if PROTOCOL.exists():raise RuntimeError('already frozen')
    manifest=load(ROOT/'config/input_manifest.json');jobs=[];inputs=[]
    main=load(manifest['main_case_path'])
    main_entry={'cohort':'main','id':'MAIN34','path':manifest['main_case_path'],'sha256':manifest['main_case_sha256'],'variants':['enhanced']}
    inputs.append(main_entry)
    for seed in range(1201,1211):jobs.append({**main_entry,'variant':'enhanced','seed':seed,'rename':False})
    witnessed=load(ROOT/'config/witnessed_input_manifest.json')['inputs']
    for entry in witnessed:
        inputs.append(entry)
        for variant in entry['variants']:
            for seed in range(5201,5211):jobs.append({**entry,'variant':variant,'seed':seed,'rename':False})
        jobs.append({**entry,'cohort':'rename','variant':'enhanced','seed':5201,'rename':True})
    for entry in manifest['inputs']:
        inputs.append(entry)
        for variant in entry['variants']:
            seeds=[1201] if entry['cohort']=='reuse' else [2201]
            for seed in seeds:jobs.append({**entry,'variant':variant,'seed':seed,'rename':False})
    protocol={'version':'process-coordinate-ga-rrp-native-ga-v2','jobs':jobs,'inputs':inputs,'total_jobs':len(jobs),
        'population':36,'generations':24,'process_population':24,'process_generations':120,'process_repair_steps':2000,
        'max_process_stages_primary':3,'max_process_stages_stress':1,
        'primary_cross_runs':570,'base_cross_runs':30,'benchmark_runs':10,'stress_runs':120,'full_rename_runs':57,
        'external_constraint_solver':False,'hybrid_pilots_excluded':True,'weights':'native 11 outer terms, native piecewise adjacency tiers mapped by route/contact priority',
        'hard_acceptance':'all safety/rotation/dimension/coverage checks + all declared main/advanced/sludge gaps + all their head margins; ancillary adjacency excluded',
        'search':'phase 1 coordinate/orientation GA plus simultaneous geometric repair and feasible-only compaction; phase 2 original AdvancedGA six-island GA/DE/SA/RL optimizing actual RRP ancillary fill around the independently generated core; feasibility-first selection',
        'global_optimum_claim':False,'physical_inputs_changed':False,'reference_site_is_hard':False,
        'historical_holdout':'114 historical synthetic cases retained as cell-level stress tests; new 57 witnessed-feasible block-level inputs separately reported; no unseen or real-plant claim',
        'sludge_head_assumptions':'preserved from preceding frozen three-route study, not measured project data',
        'development_cases':['MAIN34','A42','A46','A49','C41','C45','C48','B32','B78','B80'],
        'witness_policy':'only SHA256 and generator audits stored in protocol; coordinates never supplied to or read by search; no OS-level filesystem isolation claim',
        'scale_scope':'new cohort facility count 30..80, one aggregate treatment block per stage; count growth includes ancillary facilities, not increased cell-level parallel graph complexity',
        'engine_hashes':{n:digest(ROOT/n) for n in ENGINES},'core_sha256':digest(CORE),
        'old_adapter_sha256':digest(ROOT.parent/'independent_cross_plant_validation_20261015/native_adapter.py')}
    save(PROTOCOL,protocol);print('Frozen',len(jobs),'pure runs')

def folder(j):return ROOT/f"runs/{j['cohort']}/{j['id']}/{j['variant']}/seed_{j['seed']}"

def verify(p):
    for name,sha in p['engine_hashes'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('frozen engine changed: '+name)
    for row in p['inputs']:
        if digest(row['path'])!=row['sha256']:raise RuntimeError('frozen input changed')
    if digest(CORE)!=p['core_sha256']:raise RuntimeError('original algorithm changed')

def execute(j):
    p=load(PROTOCOL);out=folder(j);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    if (out/'result.json').exists():return {'status':'existing','id':j['id']}
    ins=load(j['path']);mapping=None
    if j['rename']:
        mapping={f['id']:f'opaque_{len(ins["facilities"])-i:04d}' for i,f in enumerate(ins['facilities'])};ins=rename_instance(ins,mapping)
    try:
        with (out/'stdout.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            stage_count=p['max_process_stages_primary'] if j['cohort'] in ('witnessed','rename','main') or ins['group']=='A' else p['max_process_stages_stress']
            with gzip.open(out/'process_candidates.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as stream:
                counter=0
                def process_observe(r,_):
                    nonlocal counter
                    counter+=1
                    stream.write(json.dumps({'index':counter,'joint':r['feasible'],'loss':r['loss'],
                        'centers':r['centers'].tolist(),'rotations':r['rotations'].tolist()},allow_nan=False)+'\n')
                core=construct(ins,j['variant'],j['seed'],p['process_population'],p['process_generations'],p['process_repair_steps'],process_observe,stage_count)
            save(out/'process_initializer.json',core)
            _,ga=make_optimizer(ins,j['variant'],j['seed'],p['population'],p['generations'] if core['found'] else 1,core_layout=core['layout'])
            first=[]
            with gzip.open(out/'candidates.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as archive:
                def observe(model,layout,chrom,details):
                    if model.unique_evaluations<=p['population']:first.append({'score':details['score'],'area':max(f.x+f.get_dimensions()[0] for f in layout)*max(f.y+f.get_dimensions()[1] for f in layout)})
                    archive.write(json.dumps({'index':model.unique_evaluations,'score':details['score'],
                        'fitness':details['selection_fitness'],'joint':details['joint_feasible'],
                        'rectangles':[[f.x,f.y,*f.get_dimensions()] for f in layout],
                        'rotations':[bool(f.rotated) for f in layout]},allow_nan=False)+'\n')
                ga.observer=observe
                final,w,h=ga.optimize()
            chosen=ga.best_evaluated or ga.best_diagnostic
            rows=serialize(chosen['layout']);geometry=audit(ins,j['variant'],rows);contract=evaluate(ins,rows)
            accepted=bool(ga.best_evaluated)
            if accepted and not(geometry['spatial_feasible'] and contract['process_and_head_all']):raise RuntimeError('independent accepted output audit failed')
            result={'status':'completed','cohort':j['cohort'],'instance':j['id'],'variant':j['variant'],'seed':j['seed'],
                'n':ins['facility_count'],'group':ins['group'],'process':ins['process_family'],
                'protocol_sha256':digest(PROTOCOL),'input_sha256':j['sha256'],'accepted':accepted,
                'process_initializer_found':core['found'],'process_initializer_stats':core['stats'],
                'initial_population_best_score':min(z['score'] for z in first),'raw_score_improvement_from_initial':min(z['score'] for z in first)-chosen['score'],
                'auxiliary_generations_completed':len(ga.fitness_history),'auxiliary_search_shortened_on_core_failure':not core['found'],
                'final_layout':serialize(final) if accepted else None,'diagnostic_layout':rows if not accepted else None,
                'geometry':geometry,'contract':contract,'raw_score':chosen['score'],'selection_fitness':chosen['components']['selection_fitness'],
                'unique_evaluations':ga.unique_evaluations,'feasible_evaluations':ga.feasible_evaluations,'decoder_calls':ga.decoder_calls,
                'core_successes':ga.core_successes,'backtracking_visits':ga.backtracking_visits,'rrp_splits':ga.split_calls,
                'rrp_fallbacks':ga.fallback_count,'history':[float(v) for v in ga.fitness_history],
                'selected_chromosome':chosen['chromosome'],'runtime_s':time.perf_counter()-start,'rename_mapping':mapping}
            save(out/'result.json',result)
        return {'status':'completed','id':j['id'],'cohort':j['cohort'],'accepted':accepted,'runtime_s':result['runtime_s']}
    except Exception:
        result={'status':'exception','job':j,'protocol_sha256':digest(PROTOCOL),'traceback':traceback.format_exc()}
        save(out/'result.json',result);return result

def run(workers=8,limit=None):
    p=load(PROTOCOL);verify(p)
    jobs=[j for j in p['jobs'] if not(folder(j)/'result.json').exists()]
    if limit is not None:jobs=jobs[:limit]
    start=time.perf_counter();done=0;accepted=0;errors=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        for fut in as_completed(futures):
            r=fut.result();done+=1;accepted+=int(r.get('accepted',False));errors+=int(r['status']=='exception')
            save(ROOT/'logs/progress.json',{'done':done,'planned':len(jobs),'accepted':accepted,'exceptions':errors,'elapsed_s':time.perf_counter()-start,'latest':r})
            if done%20==0 or done==len(jobs):print(json.dumps({'done':done,'planned':len(jobs),'accepted':accepted,'exceptions':errors,'seconds':round(time.perf_counter()-start,1)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--limit',type=int);args=ap.parse_args()
    if args.action=='prepare':prepare()
    else:run(args.workers,args.limit)
