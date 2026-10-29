"""Only main/advanced/sludge adjacency; manuscript-style segment head balance.

No plant ID determines behavior. Added residuals parameters are explicit study
assumptions, frozen separately; original missing-data coverage is retained.
"""
import copy
import math
import numpy as np

ROUTES=('main_water','advanced_water','sludge')
ADVANCED_ROLES={'ultrafiltration_rack_block','uv_reactor_room','ozone_contact','biological_activated_carbon','activated_carbon_filter'}
PARAMS={'main_water':(1.08,4.,3.),'advanced_water':(1.10,4.,2.4),
        'sludge':(1.18,5.,2.2),'backwash':(1.15,3.,1.5),'chemical':(1.25,2.5,2.),'power':(1.12,2.,1.)}


def route_class(edge,facilities):
    if edge['system']=='sludge':return 'sludge'
    if edge['system']=='advanced_water':return 'advanced_water'
    if edge['system']!='main_water':return None
    if any(facilities[n]['role'] in ADVANCED_ROLES for n in (edge['source'],edge['target'])):
        return 'advanced_water'
    return 'main_water'


def augment(original):
    ins=copy.deepcopy(original)
    facilities={f['id']:f for f in ins['facilities']}
    original_pairs={(e['source'],e['target']) for e in original['hydraulic_segments']}
    existing={(e['source'],e['target']):e for e in ins['hydraulic_segments']}
    ins['process_edges']=[]
    for e in ins['contacts']:
        group=route_class(e,facilities)
        if group:
            ins['process_edges'].append({**e,'route_class':group})
    added=[]
    # Same residuals-node ground/water levels as main-case benchmark, mapped by
    # semantic role, not by fixed facility codes. These are assumptions for the
    # synthetic cases, not measured levels of another plant.
    templates={'residuals_thickening_tank':(97.5,99.2,3.5),
               'residuals_dewatering_room':(96.5,99.0,0.),
               'dewatered_residuals_storage':(95.5,95.5,0.)}
    for f in facilities.values():
        if f['id'] not in ins['elevations'] and f['role'] in templates:
            ground,water,depth=templates[f['role']]
            ins['elevations'][f['id']]={'ground':ground,'water_out':water,'water_depth':depth,
                'type':'end' if f['role']=='dewatered_residuals_storage' else 'gravity',
                'source':'ADDED STUDY ASSUMPTION: main-case residuals-node levels reused by role, not measured plant data'}
            added.append({'field':'elevations','node':f['id'],'value':ins['elevations'][f['id']]})
    segments=[]
    for edge in ins['process_edges']:
        pair=edge['source'],edge['target']
        group=edge['route_class']
        if pair in existing:
            seg=copy.deepcopy(existing[pair])
            seg['parameter_status']='original frozen synthetic parameters'
        else:
            a,b=facilities[pair[0]],facilities[pair[1]]
            if b['role']=='residuals_thickening_tank':
                pars=[.004,.50,.30,1.2]
            elif a['role']=='residuals_thickening_tank' and b['role']=='residuals_dewatering_room':
                pars=[.005,.60,.30,1.2]
            elif a['role']=='residuals_dewatering_room' and b['role']=='dewatered_residuals_storage':
                pars=[.003,.40,.25,.8]
            else:
                raise ValueError('no explicit residuals hydraulic template for role pair')
            seg={'source':pair[0],'target':pair[1],'parameters':pars,
                 'parameter_status':'ADDED STUDY ASSUMPTION: main-case segment parameters transferred by role'}
            if b['role']=='residuals_thickening_tank':
                # Main case: GL(SB)=101.5, sludge outlet=99.5: outlet 2m below GL.
                # For new settling/filter/membrane residuals streams this is a
                # disclosed synthetic port hypothesis, not water-side effluent.
                seg['source_outlet_elevation_m']=ins['elevations'][pair[0]]['ground']-2.0
                seg['outlet_source']='ADDED STUDY ASSUMPTION: residuals outlet at source ground minus 2m; main-case burial analogue, not certified for every technology'
            added.append({'field':'hydraulic_segment','value':copy.deepcopy(seg)})
        seg['route_class']=group
        seg['system']=edge['system']
        seg['originally_covered']=pair in original_pairs
        segments.append(seg)
    ins['hydraulic_segments']=segments
    ins['route_contract']={'adjacency_scope':list(ROUTES),'adjacency_hard':True,
        'threshold_rule':'keep original per-edge targets: water 8m, residuals 15m; no relaxation',
        'original_head_pairs':len(original_pairs),'required_head_pairs':len(segments),
        'added_parameters':added,'hydraulic_method':'available=outlet-inlet; required=i*actual_routed_L+local+safety; signed_margin=available-required',
        'geometry':'variable occupied envelope, no reference-site hard constraint',
        'absent_route':'not applicable; never counted as a passed route'}
    return ins


def edge_geometry(a,b,system):
    dx=max(0.,a['x_m']-b['x_m']-b['length_m'],b['x_m']-a['x_m']-a['length_m'])
    dy=max(0.,a['y_m']-b['y_m']-b['width_m'],b['y_m']-a['y_m']-a['width_m'])
    factor,terminal,weight=PARAMS.get(system,(1.15,3.,1.))
    length=factor*(dx+dy+terminal+(1.5 if dx>0 and dy>0 else 0.))
    return math.hypot(dx,dy),length


def evaluate(ins,layout):
    byid={r['id']:r for r in layout}
    adjacency=[]
    hydraulic=[]
    for e in ins['process_edges']:
        a,b=e['source'],e['target']
        if a not in byid or b not in byid:
            adjacency.append({'route':e['route_class'],'source':a,'target':b,'status':'missing facility','satisfied':False})
            continue
        distance,_=edge_geometry(byid[a],byid[b],e['system'])
        excess=max(0.,distance-e['target_gap_m'])
        adjacency.append({'route':e['route_class'],'source':a,'target':b,'gap_m':distance,
            'threshold_m':e['target_gap_m'],'excess_m':excess,'weight':e['adjacency_weight'],
            'satisfied':excess<=1e-6,'status':'assessed'})
    for seg in ins['hydraulic_segments']:
        a,b=seg['source'],seg['target']
        if any(n not in byid or n not in ins['elevations'] for n in (a,b)):
            hydraulic.append({'route':seg['route_class'],'source':a,'target':b,'status':'missing data','feasible':False})
            continue
        up,down=ins['elevations'][a],ins['elevations'][b]
        outlet=seg.get('source_outlet_elevation_m',up['water_out'])
        inlet=seg.get('target_inlet_elevation_m',down['ground']+(-.5 if down['type']=='pump' else .30))
        _,length=edge_geometry(byid[a],byid[b],seg['system'])
        slope,local,safety,weight=seg['parameters']
        required=slope*length+local+safety
        available=outlet-inlet
        margin=available-required
        hydraulic.append({'route':seg['route_class'],'source':a,'target':b,'outlet_m':outlet,'inlet_m':inlet,
            'length_m':length,'friction_slope':slope,'friction_loss_m':slope*length,
            'local_loss_m':local,'safety_allowance_m':safety,'required_head_m':required,
            'available_head_m':available,'margin_m':margin,'surplus_m':max(0.,margin),
            'shortage_m':max(0.,-margin),'feasible':margin>=-1e-7,'status':'assessed',
            'originally_covered':seg['originally_covered'],'parameter_status':seg['parameter_status'],
            'water_level_monotone':outlet>=down['water_out']-1e-7})
    routes={}
    for group in ROUTES:
        adj=[r for r in adjacency if r['route']==group]
        head=[r for r in hydraulic if r['route']==group]
        routes[group]={'applicable':bool(adj),'edges':len(adj),'adjacency_satisfied':sum(r['satisfied'] for r in adj),
            'adjacency_all':all(r['satisfied'] for r in adj) if adj else None,
            'head_segments':len(head),'head_covered':sum(r['status']=='assessed' for r in head),
            'head_all':all(r['status']=='assessed' and r['feasible'] for r in head) if adj and len(head)==len(adj) else None,
            'minimum_margin_m':min((r['margin_m'] for r in head if r['status']=='assessed'),default=None)}
    adjacency_all=bool(adjacency) and all(r['satisfied'] for r in adjacency)
    head_complete=len(hydraulic)==len(adjacency) and all(r['status']=='assessed' for r in hydraulic)
    head_all=head_complete and all(r['feasible'] for r in hydraulic)
    return {'adjacency':adjacency,'hydraulics':hydraulic,'routes':routes,
            'adjacency_all':adjacency_all,'adjacency_rate':sum(r['satisfied'] for r in adjacency)/len(adjacency),
            'adjacency_violations':sum(not r['satisfied'] for r in adjacency),
            'adjacency_excess_m':sum(r.get('excess_m',0.) for r in adjacency),
            'head_complete':head_complete,'head_all':head_all,
            'head_violations':sum(not r['feasible'] for r in hydraulic),
            'head_shortage_m':sum(r.get('shortage_m',0.) for r in hydraulic),
            'minimum_head_margin_m':min((r['margin_m'] for r in hydraulic if r['status']=='assessed'),default=None),
            'process_and_head_all':adjacency_all and head_all}
