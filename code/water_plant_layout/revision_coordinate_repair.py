"""Random-coordinate baseline representation with deterministic nearest repair.

No residual rectangle subtraction, layout scoring, or process-name rules are
used during repair. Only nonnegative coordinates and pairwise safety clearance.
"""
from math import isfinite, sqrt
from revision_decoders import EPS


def repair_coordinates(facilities, chrom, preferred, gap_fn):
    n = len(facilities)
    if sorted(chrom['order']) != list(range(n)) or len(chrom['rotations']) != n or len(preferred) != n:
        raise ValueError('invalid chromosome')
    if any(len(p) != 2 or not all(isfinite(x) for x in p) for p in preferred):
        raise ValueError('finite coordinate pairs required')
    placed = []
    for i in chrom['order']:
        f = facilities[i].copy()
        f.rotated = bool(chrom['rotations'][i])
        w, h = f.get_dimensions()
        if not all(isfinite(v) and v > 0 for v in (w, h)):
            raise ValueError('invalid dimensions')
        px, py = max(0.0, float(preferred[i][0])), max(0.0, float(preferred[i][1]))
        xs, ys, obstacles = {0.0, px}, {0.0, py}, []
        for p in placed:
            pw, ph = p.get_dimensions()
            g = gap_fn(p, f)
            if not isfinite(g) or g < 0:
                raise ValueError('invalid safety gap')
            left, right = p.x-g-w, p.x+pw+g
            bottom, top = p.y-g-h, p.y+ph+g
            xs.update((max(0, left), right))
            ys.update((max(0, bottom), top))
            obstacles.append((left, right, bottom, top))
        candidates = []
        for x in xs:
            for y in ys:
                if all(x <= left+EPS or x >= right-EPS or y <= bottom+EPS or y >= top-EPS
                       for left, right, bottom, top in obstacles):
                    candidates.append(((x-px)**2+(y-py)**2, y, x))
        if not candidates:
            raise RuntimeError('unbounded repair should always have an exterior candidate')
        _, f.y, f.x = min(candidates)
        placed.append(f)
    if not placed:
        raise ValueError('nonempty facilities required')
    # Translation preserves all constraints and physical pair metrics.
    left, bottom = min(f.x for f in placed), min(f.y for f in placed)
    for f in placed:
        f.x -= left
        f.y -= bottom
    return placed, max(f.x+f.get_dimensions()[0] for f in placed), max(f.y+f.get_dimensions()[1] for f in placed)


def coordinate_gene_scale(facilities):
    """Declared numerical gene range; repair may extend beyond it.

    This is not a land boundary. Its sensitivity must be tested for publication.
    """
    return 1.5*sqrt(sum(f.length*f.width for f in facilities))
