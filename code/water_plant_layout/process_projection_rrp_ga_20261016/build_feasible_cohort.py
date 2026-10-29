"""Independent, witnessed-feasible facility-block validation cohort.

One aggregate civil-treatment block per process stage, matching the benchmark's
facility granularity. Increasing facility count also increases ancillary blocks;
it does not claim increasing parallel-cell graph complexity. Witness coordinates
are written ONLY to audit/generator_witnesses, never to optimizer input files.
The generator never imports or runs the search engine.
"""
import copy
import math
import numpy as np
from common import ROOT,load,save,digest
from process_model import evaluate,route_class
from validation import audit

FAMILIES={
 'daf':[('rapid_mix',.0083,'conventional'),('flocculation',.0833,'conventional'),('dissolved_air_flotation',.10,'conventional'),('rapid_granular_filter',1/7,'conventional'),('uv_reactor_room',.005,'advanced')],
 'uf':[('inline_coagulation',.0083,'conventional'),('ultrafiltration_rack_block',1/15,'advanced'),('uv_reactor_room',.005,'advanced')],
 'slow_sand':[('roughing_filter',.5,'pretreat'),('slow_sand_filter',5.,'conventional'),('uv_reactor_room',.005,'advanced')],
 'conventional_uv':[('rapid_mix',.0083,'conventional'),('flocculation',.0833,'conventional'),('sedimentation',1/1.5,'conventional'),('rapid_granular_filter',1/7,'conventional'),('uv_reactor_room',.005,'advanced')],
}
AUX=[('chemical_dosing_room','chemical',2),('disinfection_chemical_room','chemical',3),('chemical_store','chemical',2),
 ('backwash_tank','distribution',1),('backwash_pump_room','distribution',1),('source_pump_room','intake',1),
 ('power_distribution_room','power',2),('transformer_room','power',3),('control_room','admin',1),
 ('administration_block','admin',1),('gatehouse','admin',1),('laboratory','admin',1),
 ('equipment_store','auxiliary',1),('maintenance_workshop','auxiliary',1),('parking_area','auxiliary',1),
 ('staff_accommodation','living',1),('staff_service_block','living',1),('sports_ground','living',1),
 ('fire_water_tank','auxiliary',1),('fire_pump_room','auxiliary',1),('standby_generator_room','power',2),
 ('instrument_workshop','auxiliary',1),('water_quality_lab','admin',1),('metering_room','distribution',1),
 ('maintenance_store','auxiliary',1),('sampling_station','admin',1),('rainwater_pump_room','auxiliary',1),
 ('security_block','admin',1),('process_service_building','auxiliary',1),('staff_canteen','living',1)]

def safety(ins,variant,a,b):
    c=ins['constraint_versions'][variant];matrix={tuple(sorted(r['levels'])):r['distance_m'] for r in c['safety_matrix']}
    g=matrix[tuple(sorted((a['safety_level'],b['safety_level']))) ]
    o=c.get('category_overrides',{})
    if o.get(a['id'],a['category'])==o.get(b['id'],b['category']):g=max(1.,g-1.)
    return g

def build(identifier,group,n,family,seed,scenario=None):
    rng=np.random.default_rng(seed);flow=float(rng.uniform(1500,20000) if family=='slow_sand' else rng.uniform(20000,180000))
    facility=[];unique=set();elevations={};depths={}
    def add(role,area,category,level=1,aspect=None):
        ar=float(rng.uniform(1.2,2.1) if aspect is None else aspect)
        a=round(math.sqrt(max(area,12.)*ar),3);b=round(math.sqrt(max(area,12.)/ar),3)
        while (a,b) in unique or (a,b) in {(18.,12.),(10.,6.),(25.,15.),(22.,14.),(35.,18.),(30.,16.),(20.,12.),(35.,25.)}:a=round(a+.001,3)
        unique.add((a,b));name=f'node_{len(facility):03d}_{int(rng.integers(100,999))}'
        f=dict(id=name,label=role,role=role,category=category,safety_level=level,length_m=a,width_m=b,
               parameter_source='frozen capacity/area rule with independent seeded dimensions; synthetic facility block')
        facility.append(f);return f
    intake=add('intake_pump_station',float(rng.uniform(140,240)),'intake');screen=add('screen_house',float(rng.uniform(50,100)),'intake')
    chain=[intake,screen]
    for role,coef,cat in FAMILIES[family]:chain.append(add(role,flow/24*coef*float(rng.uniform(.85,1.15)),cat,2 if role=='uv_reactor_room' else 1))
    cwt=add('clear_water_reservoir',flow/24*.8*float(rng.uniform(.8,1.2)),'distribution')
    pump=add('secondary_pump_room',float(rng.uniform(140,280)),'distribution');chain.extend([cwt,pump])
    residual=next((f for f in chain if f['role'] in ('sedimentation','dissolved_air_flotation','ultrafiltration_rack_block','roughing_filter')),None)
    sludge=[]
    sludge_height=float(rng.uniform(10.,14.))
    for role in ('residuals_thickening_tank','residuals_dewatering_room','dewatered_residuals_storage'):
        h=sludge_height*float(rng.uniform(.94,1.06));a=float(rng.uniform(16.,25.));sludge.append(add(role,a*h,'sludge',aspect=a/h))
    auxiliary=[]
    while len(facility)<n:
        role,cat,level=AUX[(len(auxiliary))%len(AUX)]
        auxiliary.append(add(role,float(rng.uniform(35.,380))*(flow/50000.)**.12,cat,level))
    if len(facility)!=n:raise ValueError('facility count too small')
    for i,f in enumerate(chain):
        out=112.-i*1.6+float(seed%31)*.003
        elevations[f['id']]={'ground':out-1.1,'water_out':out,'water_depth':3.5,'type':'pump' if f in (intake,pump) else 'gravity',
            'source':'explicit synthetic water-level assumption, same segment-balance method as benchmark'}
    source_ground=elevations[residual['id']]['ground']
    thick_gl=source_ground-3.8
    for i,f in enumerate(sludge):
        gl=thick_gl-i;out=gl+(1.7 if i==0 else 2.4 if i==1 else 0.)
        elevations[f['id']]={'ground':gl,'water_out':out,'water_depth':3.5 if i==0 else 0.,'type':'gravity' if i<2 else 'end',
            'source':'explicit synthetic residuals-node level assumption; not measured project elevations'}
    contacts=[];segments=[]
    def connect(a,b,system,route=True):
        edge={'source':a['id'],'target':b['id'],'system':system,'adjacency_weight':15 if system!='sludge' else 8,'target_gap_m':15. if system=='sludge' else 8.}
        if route:
            edge['route_class']=route_class(edge,{f['id']:f for f in facility})
            if system=='sludge':
                pars=([.004,.5,.3,1.2] if a==residual else [.005,.6,.3,1.2] if a==sludge[0] else [.003,.4,.25,.8])
            else:pars=[.001,.30,.20,2.]
            seg={**edge,'parameters':pars,'originally_covered':True,'parameter_status':'frozen explicit synthetic assumptions before search'}
            if a==residual and system=='sludge':seg['source_outlet_elevation_m']=source_ground-2.
            segments.append(seg)
        contacts.append(edge)
    for a,b in zip(chain[:-1],chain[1:]):connect(a,b,'main_water')
    connect(residual,sludge[0],'sludge');connect(sludge[0],sludge[1],'sludge');connect(sludge[1],sludge[2],'sludge')
    process_edges=copy.deepcopy(contacts)
    # All ancillary links remain soft, with explicit category-directed targets.
    for f in auxiliary:
        target=(cwt if f['category']=='distribution' else chain[2] if f['category']=='chemical' else pump if f['category']=='power' else auxiliary[0] if f!=auxiliary[0] else screen)
        contacts.append({'source':f['id'],'target':target['id'],'system':'chemical' if f['category']=='chemical' else 'power' if f['category']=='power' else 'backwash' if f['category']=='distribution' else 'auxiliary',
                         'adjacency_weight':4,'target_gap_m':15.})
    baseline={(1,1):2.,(1,2):3.,(1,3):5.,(2,2):4.,(2,3):6.,(3,3):8.}
    enhanced=baseline.copy();base_override={}
    if scenario=='chemical_separation':enhanced={k:v+(2. if max(k)>=2 else 0.) for k,v in baseline.items()}
    elif scenario=='maintenance_clearance':enhanced={k:max(4.,v) for k,v in baseline.items()}
    elif scenario=='functional_zoning':base_override={f['id']:'treatment' for f in chain if f['category'] in ('conventional','advanced')}
    versions={'enhanced':{'safety_matrix':[{'levels':list(k),'distance_m':v} for k,v in enhanced.items()],'category_overrides':{},'definition':scenario or 'baseline safety and soft zoning'}}
    if scenario:versions['base']={'safety_matrix':[{'levels':list(k),'distance_m':v} for k,v in baseline.items()],'category_overrides':base_override,'definition':'matched base control'}
    ins={'id':identifier,'group':group,'facility_count':n,'process_family':family,'generation_seed':seed,'design_flow_m3_day':flow,
         'data_status':'independent synthetic facility-block cohort, witnessed feasible; no real-plant claim',
         'process_granularity':'one aggregate treatment block per stage; facility-count scaling includes ancillary blocks, not increasing parallel-cell graph complexity',
         'facilities':facility,'contacts':contacts,'process_edges':process_edges,'hydraulic_segments':segments,'elevations':elevations,
         'constraint_versions':versions,'constraint_scenario':scenario,'witness_coordinates_available_to_optimizer':False}
    # Independent shelf/branch witness, never an optimization hint.
    rows=[];x=0.;prev=None
    def row(f,x,y):return dict(id=f['id'],x_m=float(x),y_m=float(y),length_m=f['length_m'],width_m=f['width_m'],rotated=False,category=f['category'],safety_level=f['safety_level'])
    for f in chain:
        if prev:x+=safety(ins,'enhanced',prev,f)+.25
        rows.append(row(f,x,0.));x+=f['length_m'];prev=f
    byid={r['id']:r for r in rows};gmax=max(safety(ins,'enhanced',a,b) for a in chain+sludge for b in chain+sludge)
    y=-max(f['width_m'] for f in sludge)-gmax-.25;x=byid[residual['id']]['x_m'];prev=None
    for f in sludge:
        if prev:x+=safety(ins,'enhanced',prev,f)+.25
        rows.append(row(f,x,y));x+=f['length_m'];prev=f
    gap=max(enhanced.values())+1.;y=max(f['width_m'] for f in chain)+gap;x=0.
    for f in auxiliary:rows.append(row(f,x,y));x+=f['length_m']+gap
    left=min(r['x_m'] for r in rows);bottom=min(r['y_m'] for r in rows)
    for r in rows:r['x_m']-=left;r['y_m']-=bottom
    return ins,rows

def main():
    destination=ROOT/'config/witnessed_input_manifest.json'
    if destination.exists():raise RuntimeError('cohort already frozen; do not overwrite')
    specs=[('W_A42','A',42,'daf',8600101,None),('W_A46','A',46,'uf',8600203,None),('W_A49','A',49,'slow_sand',8600307,None)]
    families=list(FAMILIES)
    for n in range(30,81):specs.append((f'W_B{n}','B',n,families[(n-30)%4],8610000+n*137,None))
    for name,n,family,seed,scenario in [('W_C41',41,'daf',8620101,'chemical_separation'),('W_C45',45,'uf',8620203,'maintenance_clearance'),('W_C48',48,'slow_sand',8620307,'functional_zoning')]:specs.append((name,'C',n,family,seed,scenario))
    entries=[];proofs=[]
    for identifier,group,n,family,seed,scenario in specs:
        ins,rows=build(identifier,group,n,family,seed,scenario)
        for variant in ins['constraint_versions']:
            r=copy.deepcopy(rows);o=ins['constraint_versions'][variant].get('category_overrides',{})
            for t in r:t['category']=o.get(t['id'],t['category'])
            g=audit(ins,variant,r);p=evaluate(ins,r)
            if not(g['spatial_feasible'] and p['process_and_head_all']):raise RuntimeError((identifier,variant,g,p))
            proofs.append({'id':identifier,'variant':variant,'geometry':g,'process':p})
        path=ROOT/f'data/witnessed/{identifier}.json';save(path,ins)
        proof=ROOT/f'audit/generator_witnesses/{identifier}.json';save(proof,{'layout':rows,'purpose':'generation feasibility certificate only; search forbidden to read this file'})
        entries.append({'cohort':'witnessed','id':identifier,'path':str(path),'sha256':digest(path),'group':group,'n':n,'variants':list(ins['constraint_versions']),'witness_sha256':digest(proof)})
    save(ROOT/'audit/generator_witness_checks.json',proofs)
    save(destination,{'inputs':entries,'generator_sha256':digest(__file__),'all_witnesses_joint_feasible':True,
         'search_has_no_access_to_witness_coordinates':True,'scope':'facility-block granularity consistent with benchmark; parallel-cell stress cases reported separately'})
    print('Frozen 57 new parameter instances and 60 verified feasibility witnesses; no search performed')

if __name__=='__main__':main()
