"""CP-SAT reference for area + rotation + pairwise axis safety separation.

No adjacency, pipeline, hydraulic, zoning, road or aspect-ratio requirements.
Integral input data are required exactly, not silently rounded to a grid.
"""
import math
from ortools.sat.python import cp_model
from revision_decoders import validate_layout


def integer(value):
    if not math.isfinite(value) or value != int(value):
        raise ValueError('exact integer dimensions, clearances and hint coordinates required')
    return int(value)


def normalize_hint(layout):
    hint = [f.copy() for f in layout]
    left, bottom = min(f.x for f in hint), min(f.y for f in hint)
    for f in hint:
        f.x -= left
        f.y -= bottom
    w = max(f.x+f.get_dimensions()[0] for f in hint)
    h = max(f.y+f.get_dimensions()[1] for f in hint)
    if h > w:
        for f in hint:
            f.x, f.y = f.y, f.x
            f.rotated = not f.rotated
        w, h = h, w
    return hint, integer(w), integer(h)


def necessary_area_bound(dimensions, gap, upper):
    """Minimum integer W*H satisfying footprint and uniform-gap packing bounds.

    Inflating each side length by g_min yields disjoint rectangles in a
    (W+g_min)*(H+g_min) enclosure. This is a necessary, not sufficient condition.
    """
    footprint = sum(a*b for a,b in dimensions)
    inflated = sum((a+gap)*(b+gap) for a,b in dimensions)
    min_h = max(min(d) for d in dimensions)
    min_w = max(max(d) for d in dimensions)
    bounds = []
    for h in range(min_h, math.isqrt(upper)+1):
        w = max(h, min_w, (footprint+h-1)//h,
                (inflated-gap*h-gap*gap+h+gap-1)//(h+gap))
        if w*h <= upper:
            bounds.append(w*h)
    if not bounds:
        raise ValueError('valid hint contradicts necessary packing bound')
    return min(bounds)


def build_model(facilities, hint_layout, gap_fn):
    if not facilities or len({f.name for f in facilities}) != len(facilities):
        raise ValueError('nonempty uniquely named facility set required')
    if validate_layout(hint_layout, facilities, None, gap_fn):
        raise ValueError('a verified complete feasible hint is required')
    dimensions = [(integer(f.length), integer(f.width)) for f in facilities]
    if any(min(d) <= 0 for d in dimensions):
        raise ValueError('positive facility dimensions required')
    gaps = {(i,j): integer(gap_fn(a,b)) for i,a in enumerate(facilities) for j,b in enumerate(facilities) if i < j}
    if any(g < 0 for g in gaps.values()):
        raise ValueError('nonnegative clearance required')
    hint, hw, hh = normalize_hint(hint_layout)
    hint_by_name = {f.name: f for f in hint}
    for f in hint:
        integer(f.x)
        integer(f.y)
    upper = hw*hh
    footprint = sum(a*b for a,b in dimensions)
    min_h = max(min(d) for d in dimensions)
    min_w = max(max(d) for d in dimensions)
    max_w, max_h = upper//min_h, math.isqrt(upper)
    uniform_gap = min(gaps.values(), default=0)
    lower = necessary_area_bound(dimensions, uniform_gap, upper)
    m = cp_model.CpModel()
    W = m.new_int_var(min_w, max_w, 'W')
    H = m.new_int_var(min_h, max_h, 'H')
    area = m.new_int_var(lower, upper, 'area')
    m.add(W >= H)  # valid because the reduced model is globally transpose-invariant
    m.add_multiplication_equality(area, [W,H])
    xs, ys, widths, heights, rotations, ix, iy = [], [], [], [], [], [], []
    for i, f in enumerate(facilities):
        a,b = dimensions[i]
        x = m.new_int_var(0, max_w, f'x_{i}')
        y = m.new_int_var(0, max_h, f'y_{i}')
        r = m.new_bool_var(f'rotate_{i}')
        w = m.new_int_var(min(a,b), max(a,b), f'w_{i}')
        h = m.new_int_var(min(a,b), max(a,b), f'h_{i}')
        m.add(w == a+(b-a)*r)
        m.add(h == b+(a-b)*r)
        m.add(x+w <= W)
        m.add(y+h <= H)
        ex = m.new_int_var(0, max_w+uniform_gap, f'padded_ex_{i}')
        ey = m.new_int_var(0, max_h+uniform_gap, f'padded_ey_{i}')
        ix.append(m.new_interval_var(x, w+uniform_gap, ex, f'ix_{i}'))
        iy.append(m.new_interval_var(y, h+uniform_gap, ey, f'iy_{i}'))
        xs.append(x); ys.append(y); widths.append(w); heights.append(h); rotations.append(r)
        hf = hint_by_name[f.name]
        fw,fh = hf.get_dimensions()
        for variable,value in ((x,hf.x),(y,hf.y),(r,int(hf.rotated)),(w,fw),(h,fh),
                               (ex,hf.x+fw+uniform_gap),(ey,hf.y+fh+uniform_gap)):
            m.add_hint(variable, integer(value))
    m.add_no_overlap_2d(ix,iy)
    m.add_max_equality(W, [x+w for x,w in zip(xs,widths)])
    m.add_max_equality(H, [y+h for y,h in zip(ys,heights)])
    m.add_min_equality(0, xs)
    m.add_min_equality(0, ys)
    inflated = sum((a+uniform_gap)*(b+uniform_gap) for a,b in dimensions)
    m.add(area+uniform_gap*(W+H)+uniform_gap**2 >= inflated)
    for (i,j),g in gaps.items():
        choices = [m.new_bool_var(f'sep_{i}_{j}_{k}') for k in range(4)]
        m.add(xs[i]+widths[i]+g <= xs[j]).only_enforce_if(choices[0])
        m.add(xs[j]+widths[j]+g <= xs[i]).only_enforce_if(choices[1])
        m.add(ys[i]+heights[i]+g <= ys[j]).only_enforce_if(choices[2])
        m.add(ys[j]+heights[j]+g <= ys[i]).only_enforce_if(choices[3])
        m.add_exactly_one(choices)
        a, b = hint_by_name[facilities[i].name], hint_by_name[facilities[j].name]
        aw, ah = a.get_dimensions(); bw,bh = b.get_dimensions()
        valid = [a.x+aw+g <= b.x, b.x+bw+g <= a.x, a.y+ah+g <= b.y, b.y+bh+g <= a.y]
        selected = valid.index(True)
        for k,v in enumerate(choices):
            m.add_hint(v,int(k == selected))
    for v,value in ((W,hw),(H,hh),(area,upper)):
        m.add_hint(v,value)
    m.minimize(area)
    error = m.validate()
    if error:
        raise ValueError(error)
    info = {'n': len(facilities), 'pair_count': len(gaps), 'footprint_area': footprint,
            'uniform_minimum_gap': uniform_gap, 'analytical_lower_bound': lower,
            'verified_hint_upper_bound': upper, 'coordinate_domain_upper': [max_w,max_h],
            'global_transpose_symmetry': True, 'integer_data_verified': True}
    return m, {'x':xs,'y':ys,'w':widths,'h':heights,'r':rotations,'W':W,'H':H,'area':area}, info


def extract_layout(solver, variables, facilities):
    layout = []
    for i,f in enumerate(facilities):
        placed = f.copy()
        placed.x = solver.value(variables['x'][i])
        placed.y = solver.value(variables['y'][i])
        placed.rotated = bool(solver.value(variables['r'][i]))
        layout.append(placed)
    return layout
