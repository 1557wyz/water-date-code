"""No old input changes: import benchmark data and copy the frozen route study."""
from common import ROOT,WORKSPACE,load,save,digest
from graph_model import load_adapter

def main_case():
    mod=load_adapter().load_core()
    roles={'OZG':'ozone_contact','ACF':'activated_carbon_filter'}
    fs=[dict(id=n,label=label,length_m=a,width_m=b,category=cat,safety_level=s,role=roles.get(n,'benchmark_'+cat))
        for n,label,a,b,cat,s in mod.FACILITIES_DATA]
    contacts=[];segments=[]
    for system in ('main_water','advanced_water','sludge'):
        for a,b in mod.PIPELINE_SYSTEM[system]['pipes']:
            edge=dict(source=a,target=b,system=system,route_class=system,adjacency_weight=15,
                      target_gap_m=15. if system=='sludge' else 8.)
            contacts.append(edge)
            seg=dict(source=a,target=b,system=system,route_class=system,
                     parameters=list(mod.ELEV_SEGMENT_PARAMS.get((a,b),(.001,.30,.20,1.))),
                     originally_covered=True,parameter_status='original benchmark parameters; original default for CWT-PH2')
            if (a,b)==('SB','STK'):seg['source_outlet_elevation_m']=mod.SB_SLUDGE_OUTLET_ELEV
            segments.append(seg)
    process_edges=contacts.copy()
    weights={(a,b):weight for a,b,weight in mod.ADJACENCY_REQUIRED}
    declared={(e['source'],e['target']) for e in contacts}
    for system,definition in mod.PIPELINE_SYSTEM.items():
        for a,b in definition['pipes']:
            if (a,b) in declared:continue
            contacts.append(dict(source=a,target=b,system=system,adjacency_weight=weights.get((a,b),6),target_gap_m=15.))
            declared.add((a,b))
    for a,b,weight in mod.ADJACENCY_REQUIRED:
        if (a,b) not in declared:
            contacts.append(dict(source=a,target=b,system=mod.infer_pipeline_system(a,b),adjacency_weight=weight,target_gap_m=15.))
            declared.add((a,b))
    safety=[{'levels':list(k),'distance_m':v} for k,v in mod.SAFETY_DISTANCES.items()]
    return dict(id='MAIN34',group='regression',facility_count=len(fs),process_family='benchmark_conventional_ozone_carbon',
                facilities=fs,contacts=contacts,process_edges=process_edges,elevations=mod.ELEVATION_DATA,
                hydraulic_segments=segments,constraint_versions={'enhanced':{'safety_matrix':safety,'category_overrides':{}}},
                data_status='original manuscript benchmark inputs; paper layout never used as a search hint',
                source_core_sha256=digest(ROOT.parent/'water_plant_layout_v3.py'))

def prepare():
    p=ROOT/'data/main/MAIN34.json'
    if p.exists():raise RuntimeError('inputs already prepared')
    ins=main_case();save(p,ins)
    source=ROOT.parent/'process_routes_hydraulic_validation_20261016/config/frozen_evaluation.json'
    manifest=[]
    for row in load(source)['inputs']:
        d=load(row['path']);q=ROOT/f"data/{row['cohort']}/{row['id']}.json";save(q,d)
        manifest.append({**row,'source_path':row['path'],'source_sha256':digest(row['path']),
                         'path':str(q),'sha256':digest(q)})
    save(ROOT/'config/input_manifest.json',{'inputs':manifest,'main_case_path':str(p),'main_case_sha256':digest(p),
        'source_contract_sha256':digest(source),'thresholds_unchanged':True,'sludge_levels':'previous disclosed synthetic assumptions preserved',
        'reference_site_ignored':True,'no_paper_layout_search_hint':True})
    print('Prepared MAIN34 plus 114 unchanged physical/process/head inputs')

if __name__=='__main__':prepare()
