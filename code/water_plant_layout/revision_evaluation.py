"""Shared raw-layout evaluation for the revised 34-facility case.

No decoding, repair, or stochastic search occurs here. New objective definitions
are explicit and versioned; they must not be equated with legacy fitness.
"""
from dataclasses import dataclass
from math import hypot, isfinite

import water_plant_layout_v3 as model
from revision_decoders import validate_layout

VERSION = 'revision-evaluation-v2'
OBJECTIVES = ('area_m2', 'weighted_pipeline_m', 'adjacency_excess_weighted_m',
              'hydraulic_shortage_m')


@dataclass(frozen=True)
class EvaluationConfig:
    bounds: tuple | None
    adjacency_threshold_m: float = 5.0
    hydraulic_hard: bool = False

    def __post_init__(self):
        if self.bounds is not None and (len(self.bounds) != 2 or not all(isfinite(v) and v > 0 for v in self.bounds)):
            raise ValueError('finite positive bounds required')
        if not isfinite(self.adjacency_threshold_m) or self.adjacency_threshold_m <= 0:
            raise ValueError('finite positive adjacency threshold required')


def evaluate_layout(layout, facilities, config):
    """Evaluate only complete geometrically valid layouts; invalid metrics are null.

Adjacency is soft: satisfaction means distance <= threshold; objective is
sum(weight * max(0, distance-threshold)). Hydraulic shortage is reported without
rounding, surplus rewards or legacy penalty weights. It is soft by default.
"""
    errors = validate_layout(layout, facilities, config.bounds, model.get_safety_distance)
    result = {'version': VERSION, 'geometric_feasible': not errors,
              'feasible': False, 'errors': errors, 'metrics': None,
              'objectives': None, 'adjacency': [], 'hydraulics': []}
    if errors:
        return result
    fdict = {f.name: f for f in layout}
    paths = list(dict.fromkeys(model.MAIN_WATER_ELEV_PATH + model.SLUDGE_ELEV_PATH))
    required = {n for a, b, _ in model.ADJACENCY_REQUIRED for n in (a, b)}
    required.update(n for a, b in paths for n in (a, b))
    required.update(n for s in model.PIPELINE_SYSTEM.values() for a, b in s['pipes'] for n in (a, b))
    if required - fdict.keys():
        raise ValueError('case configuration references missing facilities')
    if any(n not in model.ELEVATION_DATA for pair in paths for n in pair):
        raise ValueError('missing hydraulic parameters')
    left, bottom = min(f.x for f in layout), min(f.y for f in layout)
    right = max(f.x+f.get_dimensions()[0] for f in layout)
    top = max(f.y+f.get_dimensions()[1] for f in layout)
    area = (right-left)*(top-bottom)
    _, pipeline = model.calculate_pipeline_network(layout)
    adjacency = []
    for a, b, weight in model.ADJACENCY_REQUIRED:
        f, g = fdict[a], fdict[b]
        fw, fh = f.get_dimensions()
        gw, gh = g.get_dimensions()
        dx = max(0, f.x-g.x-gw, g.x-f.x-fw)
        dy = max(0, f.y-g.y-gh, g.y-f.y-fh)
        distance = hypot(dx, dy)
        adjacency.append({'from': a, 'to': b, 'distance_m': distance, 'weight': weight,
                          'satisfied': distance <= config.adjacency_threshold_m,
                          'weighted_excess_m': weight*max(0, distance-config.adjacency_threshold_m)})
    hydraulics = []
    for a, b in paths:
        up, down = model.ELEVATION_DATA[a], model.ELEVATION_DATA[b]
        slope, local, safety, _ = model.ELEV_SEGMENT_PARAMS.get((a, b),
            (model.PIPE_HEADLOSS_PER_METER, model.PIPE_LOCAL_LOSS, model.GRAVITY_FLOW_SAFETY, 1.0))
        length = model.calculate_pipeline_segment(fdict[a], fdict[b], model.infer_pipeline_system(a, b))['routed_length_m']
        outlet = model.SB_SLUDGE_OUTLET_ELEV if (a, b) == ('SB', 'STK') else up['water_out']
        inlet = down['ground'] + (-0.5 if down['type'] == 'pump' else 0.3)
        required_head, available = slope*length+local+safety, outlet-inlet
        margin = available-required_head
        hydraulics.append({'from': a, 'to': b, 'length_m': length,
                           'required_head_m': required_head, 'available_head_m': available,
                           'margin_m': margin, 'shortage_m': max(0, -margin), 'feasible': margin >= 0})
    metrics = {'area_m2': area, 'utilization': sum(f.length*f.width for f in facilities)/area,
               'pipeline_m': pipeline['routed_length_m'],
               'weighted_pipeline_m': pipeline['weighted_length_m'],
               'adjacency_excess_weighted_m': sum(r['weighted_excess_m'] for r in adjacency),
               'adjacency_satisfaction': sum(r['satisfied'] for r in adjacency)/len(adjacency),
               'hydraulic_shortage_m': sum(r['shortage_m'] for r in hydraulics),
               'hydraulic_violating_segments': sum(not r['feasible'] for r in hydraulics),
               'minimum_head_margin_m': min(r['margin_m'] for r in hydraulics)}
    if not all(isfinite(v) for v in metrics.values()):
        raise ValueError('non-finite case metrics')
    hydraulic_ok = metrics['hydraulic_violating_segments'] == 0
    result.update(feasible=hydraulic_ok or not config.hydraulic_hard,
                  hydraulic_feasible=hydraulic_ok, metrics=metrics,
                  objectives=[metrics[name] for name in OBJECTIVES],
                  adjacency=adjacency, hydraulics=hydraulics)
    if config.hydraulic_hard and not hydraulic_ok:
        result['errors'].append('hydraulic shortage with hard hydraulic constraint')
    return result


def scalar_score(result, weights, scales):
    """Explicit positive reference scales; no estimated or per-run normalization."""
    if len(weights) != len(OBJECTIVES) or len(scales) != len(OBJECTIVES):
        raise ValueError('one weight and scale per objective required')
    if not all(isfinite(x) and x >= 0 for x in weights) or sum(weights) <= 0:
        raise ValueError('nonnegative finite weights with positive sum required')
    if not all(isfinite(x) and x > 0 for x in scales):
        raise ValueError('positive finite scales required')
    if not result['feasible']:
        return float('inf')
    return sum(w*v/s for w, v, s in zip(weights, result['objectives'], scales))
