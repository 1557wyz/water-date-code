"""Independent checks: recompute dimensions, pair gaps, all graph contacts.

This module never calls the decoder's clearance or score helpers for geometry.
Euclidean gap is the legacy success criterion; conservative axis separation is
also reported for the new decoder. No reference rectangle enters feasibility.
"""
import math
from collections import Counter
import numpy as np


def serialize(layout):
    return [{'id':f.name,'x_m':float(f.x),'y_m':float(f.y),
             'length_m':float(f.get_dimensions()[0]),'width_m':float(f.get_dimensions()[1]),
             'rotated':bool(f.rotated),'category':f.category,'safety_level':f.safety_level}
            for f in layout]


def audit(instance,variant,layout):
    rows=serialize(layout) if layout and hasattr(layout[0],'get_dimensions') else layout
    original={f['id']:f for f in instance['facilities']}
    constraints=instance['constraint_versions'][variant]
    overrides=constraints.get('category_overrides',{})
    matrix={tuple(sorted(r['levels'])):r['distance_m'] for r in constraints['safety_matrix']}
    counts=Counter(r['id'] for r in rows)
    missing=len(set(original)-set(counts))
    extras=len(set(counts)-set(original))
    duplicate=sum(v-1 for v in counts.values())
    malformed=0
    for r in rows:
        f=original.get(r['id'])
        vals=[r[k] for k in ('x_m','y_m','length_m','width_m')]
        if f is None or not all(math.isfinite(v) for v in vals) or min(vals[2:])<=0:
            malformed+=1
            continue
        expected=(f['width_m'],f['length_m']) if r['rotated'] else (f['length_m'],f['width_m'])
        if any(abs(a-b)>1e-7 for a,b in zip(expected,vals[2:])):
            malformed+=1
        if r['category']!=overrides.get(f['id'],f['category']) or r['safety_level']!=f['safety_level']:
            malformed+=1
    overlaps,safety,axis_violations,negative=0,0,0,0
    max_shortage=0.
    for i,a in enumerate(rows):
        if a['x_m'] < -1e-7 or a['y_m'] < -1e-7:
            negative+=1
        for b in rows[i+1:]:
            ax,ay,aw,ah=(a[k] for k in ('x_m','y_m','length_m','width_m'))
            bx,by,bw,bh=(b[k] for k in ('x_m','y_m','length_m','width_m'))
            dx=max(0.,max(ax,bx)-min(ax+aw,bx+bw))
            dy=max(0.,max(ay,by)-min(ay+ah,by+bh))
            overlap=min(ax+aw,bx+bw)-max(ax,bx)>1e-7 and min(ay+ah,by+bh)-max(ay,by)>1e-7
            gap=matrix[tuple(sorted((a['safety_level'],b['safety_level'])))]
            if a['category']==b['category']:
                gap=max(1.,gap-1.)
            shortage=max(0.,gap-math.hypot(dx,dy))
            overlaps+=int(overlap)
            safety+=int(shortage>1e-6)
            axis_violations+=int(not(ax+aw+gap<=bx+1e-7 or bx+bw+gap<=ax+1e-7 or
                                    ay+ah+gap<=by+1e-7 or by+bh+gap<=ay+1e-7))
            max_shortage=max(max_shortage,shortage)
    byid={r['id']:r for r in rows}
    excess,weighted_excess,satisfied,weighted_pipe,total_route_weight=0.,0.,0,0.,0.
    route_params={
        'main_water':(1.08,4.,3.),'advanced_water':(1.10,4.,2.4),
        'sludge':(1.18,5.,2.2),'backwash':(1.15,3.,1.5),
        'chemical':(1.25,2.5,2.),'power':(1.12,2.,1.)}
    def route(a,b,system):
        dx=max(0.,max(a['x_m'],b['x_m'])-min(a['x_m']+a['length_m'],b['x_m']+b['length_m']))
        dy=max(0.,max(a['y_m'],b['y_m'])-min(a['y_m']+a['width_m'],b['y_m']+b['width_m']))
        factor,terminal,weight=route_params.get(system,(1.15,3.,1.))
        return math.hypot(dx,dy),(dx+dy+terminal+(1.5 if dx>0 and dy>0 else 0.))*factor,weight
    systems={}
    for e in instance['contacts']:
        if e['source'] not in byid or e['target'] not in byid:
            continue
        dist,length,weight=route(byid[e['source']],byid[e['target']],e['system'])
        v=max(0.,dist-e['target_gap_m'])
        excess+=v
        weighted_excess+=e['adjacency_weight']*v
        satisfied+=int(v<=1e-6)
        weighted_pipe+=length*weight
        total_route_weight+=weight
        systems[(e['source'],e['target'])]=e['system']
    hydraulic_missing,hydraulic_violations,hydraulic_shortage=0,0,0.
    for e in instance['hydraulic_segments']:
        up,down=e['source'],e['target']
        if up not in byid or down not in byid:
            hydraulic_missing+=1
            continue
        a,b=instance['elevations'][up],instance['elevations'][down]
        system=systems.get((up,down),systems.get((down,up),'default'))
        _,length,_=route(byid[up],byid[down],system)
        source=e.get('source_outlet_elevation_m',a['water_out'])
        target=e.get('target_inlet_elevation_m',b['ground']+(-.5 if b['type']=='pump' else .3))
        slope,local,margin,_=e['parameters']
        shortage=max(0.,slope*length+local+margin-(source-target))
        hydraulic_violations+=int(shortage>1e-7)
        hydraulic_shortage+=shortage
    if not rows:
        return {'spatial_feasible':False,'missing':len(original)}
    width=max(r['x_m']+r['length_m'] for r in rows)-min(r['x_m'] for r in rows)
    height=max(r['y_m']+r['width_m'] for r in rows)-min(r['y_m'] for r in rows)
    footprint=sum(f['length_m']*f['width_m'] for f in instance['facilities'])
    spatial=not(missing or extras or duplicate or malformed or overlaps or safety or negative)
    # A graph target is soft, unless a future schema explicitly declares otherwise.
    return {'spatial_feasible':spatial,'missing':missing,'extra':extras,'duplicate':duplicate,
            'malformed':malformed,'overlap_pairs':overlaps,'safety_violation_pairs':safety,
            'axis_separation_violations':axis_violations,'negative':negative,
            'max_safety_shortage_m':max_shortage,'width_m':width,'height_m':height,
            'area_m2':width*height,'footprint_m2':footprint,'packing_expansion':width*height/footprint,
            'utilization':footprint/(width*height),'aspect':max(width,height)/min(width,height),
            'contacts':len(instance['contacts']),'adjacency_satisfaction':satisfied/len(instance['contacts']),
            'adjacency_excess_m':excess,'weighted_adjacency_excess_m':weighted_excess,
            'weighted_pipeline_m':weighted_pipe,'normalized_connection':weighted_pipe/(total_route_weight*math.sqrt(footprint)),
            'hydraulic_missing':hydraulic_missing,'hydraulic_violation_segments':hydraulic_violations,
            'hydraulic_shortage_m':hydraulic_shortage,
            'hydraulic_proxy_feasible':not(hydraulic_missing or hydraulic_violations)}
