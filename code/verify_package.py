"""Portable audit of the frozen supplementary evidence; never runs an optimizer."""
from pathlib import Path
import csv, json, gzip, hashlib, math, itertools, statistics, sys
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parent.parent
if not (ROOT/'data').exists():
    ROOT=Path(__file__).resolve().parent/'Supplementary_Materials'
checks=[]
def check(name,condition,detail=None):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})
    if not condition: print('FAILED:',name,detail)
def read(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def truth(x):return str(x).lower() in ('true','1')
def near(a,b,tol=1e-6):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=tol)
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gap(a,b):
    dx=max(0,max(a['x_m'],b['x_m'])-min(a['x_m']+a['length_m'],b['x_m']+b['length_m']))
    dy=max(0,max(a['y_m'],b['y_m'])-min(a['y_m']+a['width_m'],b['y_m']+b['width_m']))
    return math.hypot(dx,dy)
def contact(a,b):
    return abs(a['x_m']+a['length_m']/2-b['x_m']-b['length_m']/2)+abs(a['y_m']+a['width_m']/2-b['y_m']-b['width_m']/2)

manifest_path=ROOT/'SHA256SUMS.json'
if manifest_path.exists():
    hashes=load('SHA256SUMS.json')
    bad=[n for n,h in hashes.items() if not (ROOT/n).is_file() or hashfile(ROOT/n)!=h]
    check('All manifest file hashes',not bad,{'files':len(hashes),'mismatches':bad})
else:
    check('Final file hash manifest exists',False)

index=read('data_index.csv')
bad=[]
for x in index:
    p=ROOT/x['file']
    rows=read(x['file'])
    if len(rows)!=int(x['rows']) or hashfile(p)!=x['sha256']:bad.append(x['file'])
check('CSV index counts and hashes',not bad,{'tables':len(index),'mismatches':bad})

source_manifest=load('data/controlled/manifest.json')
check('Eight controlled source hashes',len(source_manifest['sources'])==8 and all(
    hashfile(ROOT/'code/water_plant_layout'/n)==h for n,h in source_manifest['sources'].items()))
raw=list((ROOT/'data/controlled/raw').glob('*.json.gz'))
records=[json.loads(gzip.decompress(p.read_bytes())) for p in sorted(raw)]
check('Controlled runs and evaluation count',len(records)==60 and sum(x['summary']['evaluations'] for x in records)==92160)
check('Controlled seed replication',all(v==6 for v in Counter(x['summary']['settings']['seed'] for x in records).values()) and
      set(x['summary']['settings']['seed'] for x in records)==set(range(1101,1111)))
scales=[15000,1000,10000,1]
score_errors=[]
for d in records:
    s=d['summary'];m=s['metrics'];values=[m['area_m2'],m['weighted_pipeline_m'],m['adjacency_excess_weighted_m'],m['hydraulic_shortage_m']]
    score_errors.append(abs(sum(v/c for v,c in zip(values,scales))-s['score']))
check('All controlled representative score arithmetic',max(score_errors)<1e-9,max(score_errors))
check('All 60 representative hydraulic shortages are zero',all(x['summary']['metrics']['hydraulic_shortage_m']==0 for x in records))
meta=load('data/controlled/figure3/experiment_metadata.json')
ps=[]
for c in meta['comparisons']:
    ds=c['differences'];obs=abs(sum(ds));n=len(ds)
    p=sum(abs(sum(a*b for a,b in zip(ds,signs)))>=obs-1e-12 for signs in itertools.product([-1,1],repeat=n))/(2**n)
    check('Exact paired sign flip '+c['label'],near(p,c['p_raw'],1e-12),p);ps.append(p)
order=sorted(range(len(ps)),key=lambda i:ps[i]);prev=0;adjusted=[0]*len(ps)
for rank,i in enumerate(order):
    prev=max(prev,min(1,(len(ps)-rank)*ps[i]));adjusted[i]=prev
check('Holm adjustment over three prespecified contrasts',all(near(p,c['p_holm'],1e-12) for p,c in zip(adjusted,meta['comparisons'])))

primary=read('data/synthetic/primary_runs.csv');baseline=read('data/synthetic/constraint_baseline_runs.csv')
synheads=read('data/synthetic/primary_hydraulic_segments.csv')
check('Synthetic primary denominator',len(primary)==570 and len(set(x['instance'] for x in primary))==57 and all(v==10 for v in Counter(x['instance'] for x in primary).values()))
check('Synthetic accepted primary runs',all(truth(x['accepted']) and truth(x['spatial_feasible']) and truth(x['three_routes_adjacent']) and truth(x['head_feasible']) for x in primary))
check('Synthetic extra paired baselines',len(baseline)==30 and len(set(x['instance'] for x in baseline))==3)
check('Synthetic declared head segments',len(synheads)==5670 and near(min(float(x['margin_m']) for x in synheads),.6009021675237489))
errs=[]
for x in synheads:
    errs.extend([abs(float(x['available_head_m'])-(float(x['outlet_m'])-float(x['inlet_m']))),
                 abs(float(x['required_head_m'])-(float(x['length_m'])*float(x['friction_slope'])+float(x['local_loss_m'])+float(x['safety_allowance_m']))),
                 abs(float(x['margin_m'])-(float(x['available_head_m'])-float(x['required_head_m'])))])
check('Synthetic head arithmetic on all 5670 segments',max(errs)<1e-8,max(errs))
layouts=list((ROOT/'data/synthetic/layouts').glob('*.csv'))
check('657 synthetic layout records',len(layouts)==657,{'primary_and_baseline':len([p for p in layouts if not p.name.startswith('synthetic_id_rename')]),'renaming':len([p for p in layouts if p.name.startswith('synthetic_id_rename')])})
layout_errors=[];dimension_errors=[]
for x in primary+baseline:
    path=f"data/synthetic/layouts/synthetic_{x['group']}_{x['instance']}_{x['variant']}_{x['seed']}.csv"
    rr=read(path);fac=load('data/synthetic/inputs/'+x['instance']+'.json')['facilities']
    original={a['id']:a for a in fac}
    if len(rr)!=int(x['n']) or set(a['id'] for a in rr)!=set(original):dimension_errors.append(path)
    for a in rr:
        before=sorted([original[a['id']]['length_m'],original[a['id']]['width_m']])
        after=sorted([float(a['length_m']),float(a['width_m'])])
        if not all(near(b,c) for b,c in zip(before,after)):dimension_errors.append(path)
    w=max(float(a['x_m'])+float(a['length_m']) for a in rr)-min(float(a['x_m']) for a in rr)
    h=max(float(a['y_m'])+float(a['width_m']) for a in rr)-min(float(a['y_m']) for a in rr)
    area=w*h;footprint=sum(float(a['length_m'])*float(a['width_m']) for a in rr)
    if not near(area,x['area_m2'],1e-5) or not near(footprint/area,x['utilization'],1e-8):layout_errors.append(path)
check('All 600 synthetic layouts retain input facility dimensions',not dimension_errors,{'mismatches':sorted(set(dimension_errors))})
check('All 600 synthetic enclosing areas and utilizations',not layout_errors,{'mismatches':layout_errors})
reps=read('data/synthetic/representatives_all57.csv')
bad=[]
for x in reps:
    runs=sorted([v for v in primary if v['instance']==x['instance']],key=lambda v:float(v['raw_score']))
    if int(x['seed'])!=int(runs[len(runs)//2]['seed']):bad.append(x['instance'])
check('57 upper median representative selections',len(reps)==57 and not bad,bad)

sites=read('data/real_sites/plant_summary.csv')
edges=read('data/real_sites/contact_edges.csv')
heads=read('data/real_sites/hydraulic_segments.csv')
adj=read('data/real_sites/adjacency_checks.csv')
geo={x['case']:x for x in load('data/real_sites/frozen_geometry.json')}
check('Anonymous site labels',set(geo)=={'T','L1','C','B','H1','H2','A'} and set(x['plant'] for x in sites)==set(geo))
check('Selected site denominators',len(sites)==7 and len(edges)==145 and len(heads)==78 and sum(int(x['facility_blocks']) for x in sites)==125)
errors=[];adjerrors=[];siteerrors=[]
for s in sites:
    c=s['case'];g=geo[c];src={v['id']:v for v in g['source_blocks']};dst={v['id']:v for v in g['candidate_blocks']}
    check('Preserved facility dimensions '+c,set(src)==set(dst) and all(sorted([src[i]['length_m'],src[i]['width_m']])==sorted([dst[i]['length_m'],dst[i]['width_m']]) for i in src))
    these=[x for x in edges if x['case']==c]
    for x in these:
        errors.extend([abs(contact(src[x['source']],src[x['target']])-float(x['source_contact_m'])),
                       abs(contact(dst[x['source']],dst[x['target']])-float(x['candidate_contact_m']))])
    a=sum(float(x['source_contact_m']) for x in these);b=sum(float(x['candidate_contact_m']) for x in these)
    siteerrors.extend([abs(a-float(s['source_contact_m'])),abs(b-float(s['candidate_contact_m'])),abs((b/a-1)*100-float(s['source_to_candidate_change_percent']))])
    for x in adj:
        if x['case']==c:adjerrors.append(abs(gap(dst[x['source']],dst[x['target']])-float(x['gap_m'])))
check('Independent centre Manhattan calculations on 145 contacts',max(errors)<1e-7,max(errors))
check('Contact totals and seven change percentages',max(siteerrors)<1e-7,max(siteerrors))
check('Independent Euclidean gaps on 78 process connections',max(adjerrors)<1e-7,max(adjerrors))
check('78 declared process adjacency limits',all(float(x['gap_m'])<=float(x['threshold_m'])+1e-6 for x in adj))
errs=[]
for x in heads:
    a=float(x['outlet_m'])-float(x['inlet_m']);req=float(x['estimated_external_pipe_m'])*float(x['friction_slope'])+float(x['local_loss_m'])+float(x['safety_allowance_m'])
    errs.extend([abs(a-float(x['available_head_m'])),abs(req-float(x['required_head_m'])),abs(a-req-float(x['margin_m']))])
check('Real site head arithmetic',max(errs)<1e-8,max(errs))
check('78 nonnegative real site head margins',all(float(x['margin_m'])>=-1e-7 for x in heads))
check('Site route denominators',Counter(x['system'] for x in heads)=={'main_water':39,'advanced_water':9,'sludge':30})
check('Seven declared joint passes',all(truth(s['joint_pass']) for s in sites))
t=geo['T'];dst={v['id']:v for v in t['candidate_blocks']}
pairs=read('data/real_sites/T_pair_clearances.csv');nn=read('data/real_sites/T_nearest_neighbours.csv')
check('T all 276 pairs at least 7 m',len(pairs)==276 and all(gap(dst[x['source']],dst[x['target']])>=7-1e-6 and near(gap(dst[x['source']],dst[x['target']]),x['actual_m']) for x in pairs))
actual_nearest=[min(gap(a,b) for b in dst.values() if b['id']!=a['id']) for a in dst.values()]
check('T 24 nearest neighbours in requested range',len(nn)==24 and all(5-1e-6<=v<=10+1e-6 for v in actual_nearest),{'minimum_m':min(actual_nearest),'maximum_m':max(actual_nearest)})
check('T selected current metrics',near(next(s['candidate_contact_m'] for s in sites if s['case']=='T'),4483.772786862033) and near(min(float(x['margin_m']) for x in heads),.47601171623769933))
reductions=[-float(s['source_to_candidate_change_percent']) for s in sites]
check('Seven site median contact reduction',near(statistics.median(reductions),32.389352208777886),statistics.median(reductions))

# Optional independent polygon calculations, beyond the supplied diagnostic records.
try:
    from shapely.geometry import shape,box
    from shapely.ops import unary_union
except ImportError:
    checks.append({'check':'Optional Shapely land and road audit','passed':None,'detail':'Not run; install Shapely to enable. All arithmetic and rectangle checks above remain active.'})
else:
    landerrors=[]
    for c,g in geo.items():
        land=g['candidate_land_geometry'];site=shape(g['site']);fac=[box(a['x_m'],a['y_m'],a['x_m']+a['length_m'],a['y_m']+a['width_m']) for a in g['candidate_blocks']]
        union=unary_union(fac)
        err={'site':c,'outside_m2':union.difference(site).area,'facility_overlap_m2':sum(v.area for v in fac)-union.area}
        for key in ['roads','reserve','fixed_no_build']:
            if key in land:err[key+'_overlap_m2']=union.intersection(shape(land[key])).area
        road=shape(land['roads']);err['max_access_distance_m']=max(v.distance(road) for v in fac)
        landerrors.append(err)
    check('Independent polygon land and road audit',all(all(v<1e-5 for k,v in e.items() if k!='site') for e in landerrors),landerrors)

report={'status':'PASS' if all(c['passed'] is not False for c in checks) else 'FAIL',
        'checks_passed':sum(c['passed'] is True for c in checks),'checks_failed':sum(c['passed'] is False for c in checks),
        'checks_optional_not_run':sum(c['passed'] is None for c in checks),'checks':checks}
out=ROOT/'verification/package_verification.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
sys.exit(0 if report['status']=='PASS' else 1)
