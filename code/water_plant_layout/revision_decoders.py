"""Experimental fixed-boundary decoders; does not change the legacy optimizer.

RRP rebuilds free rectangles for each item because safety gaps depend on the
pair of facilities. Rectangles can overlap; contained rectangles are pruned.
Only order and rotations are active genes. Boundary failure is explicit.
"""
from dataclasses import dataclass
from math import isfinite

EPS = 1e-9


def computational_bounds(facilities, gap_fn):
    """Conservative row/column construction, not an approved-site constraint.

    Each orientation fits within its max dimension. Summing those dimensions
    and n-1 maximum gaps bounds a complete single-row layout in either axis.
    """
    if not facilities:
        raise ValueError('nonempty facilities required')
    dimensions = [max(f.get_dimensions()) for f in facilities]
    gaps = [gap_fn(a, b) for i, a in enumerate(facilities) for b in facilities[i+1:]]
    if not all(isfinite(d) and d > 0 for d in dimensions) or not all(isfinite(g) and g >= 0 for g in gaps):
        raise ValueError('invalid dimensions or gaps')
    side = sum(dimensions) + (len(facilities)-1)*max(gaps, default=0)
    return (side, side)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def subtract(free, obstacle):
    """Split all intersected free rectangles into up to four residual strips."""
    out = []
    for r in free:
        left, bottom = max(r.x, obstacle.x), max(r.y, obstacle.y)
        right = min(r.x + r.w, obstacle.x + obstacle.w)
        top = min(r.y + r.h, obstacle.y + obstacle.h)
        if right <= left + EPS or top <= bottom + EPS:
            out.append(r)
            continue
        out.extend((Rect(r.x, r.y, left-r.x, r.h),
                    Rect(right, r.y, r.x+r.w-right, r.h),
                    Rect(r.x, r.y, r.w, bottom-r.y),
                    Rect(r.x, top, r.w, r.y+r.h-top)))
    unique = sorted(set(r for r in out if r.w > EPS and r.h > EPS),
                    key=lambda r: (r.x, r.y, r.w, r.h))
    return [r for i, r in enumerate(unique) if not any(
        i != j and s.x <= r.x + EPS and s.y <= r.y + EPS
        and s.x+s.w >= r.x+r.w-EPS and s.y+s.h >= r.y+r.h-EPS
        for j, s in enumerate(unique))]


def validate_layout(placed, facilities, bounds, gap_fn):
    """Independent geometry audit, using the manuscript's axis separation rule."""
    errors = []
    names = [f.name for f in placed]
    expected = [f.name for f in facilities]
    if sorted(names) != sorted(expected):
        errors.append('missing, duplicate or unexpected facilities')
    original = {f.name: f for f in facilities}
    for f in placed:
        w, h = f.get_dimensions()
        if not all(isfinite(v) for v in (f.x, f.y, w, h)) or w <= 0 or h <= 0:
            errors.append(f'{f.name}: invalid geometry')
        if f.name in original and (f.length, f.width) != (original[f.name].length, original[f.name].width):
            errors.append(f'{f.name}: changed dimensions')
        if f.name in original and (f.category, f.safety_level) != (original[f.name].category, original[f.name].safety_level):
            errors.append(f'{f.name}: changed safety metadata')
        if f.x < -EPS or f.y < -EPS or (bounds is not None and
                (f.x+w > bounds[0]+EPS or f.y+h > bounds[1]+EPS)):
            errors.append(f'{f.name}: boundary violation')
    for i, a in enumerate(placed):
        aw, ah = a.get_dimensions()
        for b in placed[i+1:]:
            bw, bh = b.get_dimensions()
            gap = gap_fn(a, b)
            if not (a.x+aw+gap <= b.x+EPS or b.x+bw+gap <= a.x+EPS
                    or a.y+ah+gap <= b.y+EPS or b.y+bh+gap <= a.y+EPS):
                errors.append(f'{a.name}/{b.name}: overlap or safety violation')
    return errors


def decode(facilities, chrom, bounds, gap_fn, mode='rrp'):
    """Return (placed, occupied_width, occupied_height), possibly partial on failure.

Both modes use identical order, rotations, bounds, clearance checks and
area/y/x tie breaking. The edge mode is a deterministic candidate-placement
control, NOT yet a standard GA or a random-initialization-and-repair baseline.
"""
    if mode not in ('rrp', 'edge'):
        raise ValueError('mode must be rrp or edge')
    if bounds is None:
        bounds = computational_bounds(facilities, gap_fn)
    if len(bounds) != 2 or not all(isfinite(v) and v > 0 for v in bounds):
        raise ValueError('positive finite fixed bounds required')
    n = len(facilities)
    if sorted(chrom['order']) != list(range(n)) or len(chrom['rotations']) != n:
        raise ValueError('invalid chromosome')
    if len({f.name for f in facilities}) != n:
        raise ValueError('facility names must be unique')
    placed, max_x, max_y = [], 0.0, 0.0
    for idx in chrom['order']:
        f = facilities[idx].copy()
        f.rotated = bool(chrom['rotations'][idx])
        w, h = f.get_dimensions()
        if not all(isfinite(v) and v > 0 for v in (w, h)):
            raise ValueError('invalid facility dimensions')
        obstacles = []
        for p in placed:
            gap = gap_fn(p, f)
            if not isfinite(gap) or gap < 0:
                raise ValueError('invalid safety distance')
            pw, ph = p.get_dimensions()
            obstacles.append(Rect(p.x-gap, p.y-gap, pw+2*gap, ph+2*gap))
        if mode == 'rrp':
            free = [Rect(0, 0, *bounds)]
            for obstacle in obstacles:
                free = subtract(free, obstacle)
            candidates = {(r.x, r.y) for r in free if r.w+EPS >= w and r.h+EPS >= h}
        else:
            xs = {0.0} | {r.x+r.w for r in obstacles}
            ys = {0.0} | {r.y+r.h for r in obstacles}
            candidates = {(x, y) for x in xs for y in ys}
        valid = []
        for x, y in candidates:
            if x < -EPS or y < -EPS or x+w > bounds[0]+EPS or y+h > bounds[1]+EPS:
                continue
            if any(not (x+w <= r.x+EPS or x >= r.x+r.w-EPS
                        or y+h <= r.y+EPS or y >= r.y+r.h-EPS) for r in obstacles):
                continue
            valid.append((max(max_x, x+w)*max(max_y, y+h), y, x))
        if not valid:
            break
        _, f.y, f.x = min(valid)
        placed.append(f)
        max_x, max_y = max(max_x, f.x+w), max(max_y, f.y+h)
    return placed, max_x, max_y


def attach_decoder(ga, bounds, mode='rrp'):
    """Opt-in adapter for AdvancedGA, with a fixed boundary at all call sites.

For decoder checks and calculate_fitness only. Legacy optimize() includes
postprocessing and unresolved RL issues; it is not a validated experiment yet.
"""
    from types import MethodType
    from water_plant_layout_v3 import get_safety_distance
    def decode_fixed(self, chrom, container_size):
        return decode(self.facilities, self._repair_chromosome(chrom),
                      bounds, get_safety_distance, mode)
    ga.decode_chromosome = MethodType(decode_fixed, ga)
    ga._fitness_cache.clear()
    return ga
