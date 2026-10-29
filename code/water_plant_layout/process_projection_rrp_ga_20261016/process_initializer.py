"""Process-coordinate GA initializer; input-only and no witness access."""
import numpy as np
from evolve_core import initialize_core
from geometric_repair import relax

def prepare_problem(ins,variant):
    nodes={e[k] for e in ins['process_edges'] for k in ('source','target')}
    fs=[f for f in ins['facilities'] if f['id'] in nodes];idx={f['id']:i for i,f in enumerate(fs)};n=len(fs)
    sizes=np.array([[f['length_m'],f['width_m']] for f in fs],dtype=float)
    c=ins['constraint_versions'][variant];override=c.get('category_overrides',{})
    matrix={tuple(sorted(r['levels'])):r['distance_m'] for r in c['safety_matrix']}
    gaps=np.zeros((n,n));targets=np.full((n,n),-1.);head=np.full((n,n),np.inf)
    for i,a in enumerate(fs):
        for j,b in enumerate(fs):
            gap=matrix[tuple(sorted((a['safety_level'],b['safety_level']))) ]
            if override.get(a['id'],a['category'])==override.get(b['id'],b['category']):gap=max(1.,gap-1.)
            gaps[i,j]=gap
    for e in ins['process_edges']:
        i,j=idx[e['source']],idx[e['target']]
        t=e['target_gap_m']
        if targets[i,j]>=0:t=min(t,targets[i,j])
        targets[i,j]=targets[j,i]=t
    for e in ins['hydraulic_segments']:
        i,j=idx[e['source']],idx[e['target']];a,b=ins['elevations'][e['source']],ins['elevations'][e['target']]
        source=e.get('source_outlet_elevation_m',a['water_out']);target=e.get('target_inlet_elevation_m',b['ground']+(-.5 if b['type']=='pump' else .3))
        factor,terminal={'main_water':(1.08,4.),'advanced_water':(1.10,4.),'sludge':(1.18,5.)}[e['system']]
        slope,local,margin,_=e['parameters'];limit=(source-target-local-margin)/(slope*factor)-terminal
        head[i,j]=head[j,i]=min(head[i,j],limit)
    return fs,sizes,gaps,targets,head

def construct(ins,variant,seed,population=36,generations=180,steps=2500,observer=None,stages=1):
    fs,sizes,gaps,targets,head=prepare_problem(ins,variant)
    stage_stats=[];warm=None
    for stage in range(stages):
        best,stats=initialize_core(sizes,gaps,targets,head,seed+stage*100003,population,generations,steps,
            warm=warm,observer=observer,groups=np.array([f['role'] for f in fs]))
        stage_stats.append(stats)
        if best['feasible']:break
        oriented=sizes.copy();oriented[best['rotations']]=oriented[best['rotations']][:,::-1]
        warm=[{'rotations':best['rotations'].tolist(),'positions':(best['centers']-oriented/2).tolist()}]
    stats={'stages':stage_stats,'evaluations':sum(s['evaluations'] for s in stage_stats),'elapsed_s':sum(s['elapsed_s'] for s in stage_stats)}
    oriented=sizes.copy();oriented[best['rotations']]=oriented[best['rotations']][:,::-1]
    initial_area=float(np.prod(np.ptp(best['centers']-oriented/2,axis=0)+0.))
    # Feasible-only compaction: every trial retains the original dimensions and
    # gets rechecked by the same exact residual before it can replace the seed.
    def area(xy):
        p=xy-oriented/2
        return float(np.prod(np.max(p+oriented,axis=0)-np.min(p,axis=0)))
    initial_area=area(best['centers']);best_area=initial_area;compact_trials=0
    if best['feasible']:
        for k in range(40):
            trial=best['centers'].copy();trial[:,k%2]*=.99
            xy,ok,_,loss,bad,count=relax(trial,oriented,gaps,targets,head,250,seed+k*37+1)
            compact_trials+=1
            if ok and area(xy)<best_area:
                best['centers']=xy;best_area=area(xy);best['loss']=float(loss);best['maximum_error_m']=float(bad);best['violations']=int(count)
    stats.update(initial_core_area_m2=initial_area,compacted_core_area_m2=best_area,compaction_trials=compact_trials)
    positions=best['centers']-oriented/2;positions-=positions.min(axis=0)
    override=ins['constraint_versions'][variant].get('category_overrides',{})
    rows=[{'id':f['id'],'x_m':float(positions[i,0]),'y_m':float(positions[i,1]),
           'length_m':float(oriented[i,0]),'width_m':float(oriented[i,1]),'rotated':bool(best['rotations'][i]),
           'category':override.get(f['id'],f['category']),'safety_level':f['safety_level']} for i,f in enumerate(fs)]
    return {'found':best['feasible'],'layout':rows,'loss':best['loss'],'maximum_error_m':best['maximum_error_m'],
            'violations':best['violations'],'stats':stats,'method':'coordinate-and-orientation GA with local joint geometric repair, no external solver or witness'}
