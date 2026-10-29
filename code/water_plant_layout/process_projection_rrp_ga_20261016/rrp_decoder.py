"""Variable-envelope RRP: explicit disjoint residual rectangles, no site bounds.

The square workspace is a constructive numerical bound for a single-row packing,
not a land boundary. Reported width/height are the occupied envelope. Pair-specific
clearances require rebuilding residual space for each incoming rectangle. Strips
are disjoint; crossing adjacent strips is not explored. A constructive row fallback
is explicit and counted. No ID, label or process-name inference occurs here.
"""
import math
import numpy as np
from numba import njit

EPS = 1e-8


@njit(cache=True)
def split_rectangles(free, obstacle):
    """Exact disjoint subtraction: left/right full strips, bottom/top middle."""
    ox, oy, ow, oh = obstacle
    out = np.empty((4 * len(free), 4), dtype=np.float64)
    count = 0
    for k in range(len(free)):
        x, y, w, h = free[k]
        left, bottom = max(x, ox), max(y, oy)
        right, top = min(x+w, ox+ow), min(y+h, oy+oh)
        if right <= left+EPS or top <= bottom+EPS:
            out[count] = free[k]
            count += 1
        else:
            # These four pieces have disjoint interiors and preserve all free area.
            if left > x+EPS:
                out[count] = (x, y, left-x, h)
                count += 1
            if x+w > right+EPS:
                out[count] = (right, y, x+w-right, h)
                count += 1
            if bottom > y+EPS:
                out[count] = (left, y, right-left, bottom-y)
                count += 1
            if y+h > top+EPS:
                out[count] = (left, top, right-left, y+h-top)
                count += 1
    return out[:count].copy()


@njit(cache=True)
def pack(order, sizes, gaps, contact_weights, target_gaps, categories, strategy):
    n = len(order)
    # Accommodates either orientation, all pair gaps and a constructive row.
    side = np.sum(np.maximum(sizes[:, 0], sizes[:, 1])) + n*np.max(gaps)+1.0
    positions = np.zeros((n, 2), dtype=np.float64)
    active = np.zeros(n, dtype=np.bool_)
    max_x, max_y = 0.0, 0.0
    attempts, split_calls, fallbacks, peak_free = 0, 0, 0, 1
    trace = np.zeros((n, 5), dtype=np.float64)
    for step in range(n):
        idx = order[step]
        w, h = sizes[idx]
        free = np.array([[0.0, 0.0, side, side]])
        for t in range(step):
            j = order[t]
            gap = gaps[idx, j]
            obstacle = np.array([positions[j, 0]-gap, positions[j, 1]-gap,
                                 sizes[j, 0]+2*gap, sizes[j, 1]+2*gap])
            free = split_rectangles(free, obstacle)
            split_calls += 1
            peak_free = max(peak_free, len(free))
        best_score, bx, by = np.inf, 0.0, 0.0
        for k in range(len(free)):
            x, y, fw, fh = free[k]
            if fw+EPS < w or fh+EPS < h:
                continue
            attempts += 1
            # Independently check candidate against every placed rectangle.
            valid = True
            process, zone = 0.0, 0.0
            for t in range(step):
                j = order[t]
                px, py = positions[j]
                pw, ph = sizes[j]
                gap = gaps[idx, j]
                if not (x+w+gap <= px+EPS or px+pw+gap <= x+EPS
                        or y+h+gap <= py+EPS or py+ph+gap <= y+EPS):
                    valid = False
                    break
                dx = max(0.0, max(x, px)-min(x+w, px+pw))
                dy = max(0.0, max(y, py)-min(y+h, py+ph))
                dist = math.sqrt(dx*dx+dy*dy)
                if contact_weights[idx, j] > 0:
                    process += contact_weights[idx, j]*max(0.0, dist-target_gaps[idx, j])
                if categories[idx] == categories[j]:
                    zone += abs(x+w/2-px-pw/2)+abs(y+h/2-py-ph/2)
            if not valid:
                continue
            nw, nh = max(max_x, x+w), max(max_y, y+h)
            aspect = max(nw, nh)/max(min(nw, nh), EPS)
            # Frozen local construction heuristic, distinct from global objective.
            pweight = 6.0 if strategy == 0 else (2.0 if strategy == 1 else 1.0)
            zweight = 2.0 if strategy == 1 else 0.1
            score = nw*nh + pweight*process + zweight*zone + 20*max(0.0, aspect-3.0)
            if score < best_score-EPS or (abs(score-best_score) <= EPS and
                                         (y < by-EPS or (abs(y-by) <= EPS and x < bx))):
                best_score, bx, by = score, x, y
        if not math.isfinite(best_score):
            # Disjoint partitioning can hide a fitting rectangle spanning strips.
            # A new right-hand column is always safe within the constructive side.
            bx, by = max_x+np.max(gaps[idx]), 0.0
            if step == 0:
                bx = 0.0
            if bx+w > side+EPS or h > side+EPS:
                raise ValueError('constructive numerical workspace exhausted')
            fallbacks += 1
        positions[idx, 0], positions[idx, 1] = bx, by
        active[idx] = True
        max_x, max_y = max(max_x, bx+w), max(max_y, by+h)
        trace[step] = (float(idx), float(len(free)), bx, by, best_score if math.isfinite(best_score) else -1.0)
    return positions, max_x, max_y, attempts, split_calls, fallbacks, peak_free, trace


def graph_order(chromosome, predecessors, categories):
    """Kahn frontier ordered by genotype; process edges must form a DAG.

Precedence is a construction order only, never a claim that geometry enforces
hydraulic direction. Other systems can be cyclic and remain scored contacts.
"""
    order = chromosome['order']
    rank = {idx: k for k, idx in enumerate(order)}
    zone_rank = {c: k for k, c in enumerate(chromosome['zone_order'])}
    remaining, done, result = set(order), set(), []
    while remaining:
        ready = [i for i in order if i in remaining and predecessors[i].issubset(done)]
        if not ready:
            raise ValueError('directed treatment graph contains a cycle; declare recycle links separately')
        if chromosome['strategy'] == 1:
            idx = min(ready, key=lambda i: (zone_rank[categories[i]], rank[i]))
        else:
            idx = min(ready, key=lambda i: rank[i])
        result.append(idx)
        remaining.remove(idx)
        done.add(idx)
    return np.asarray(result, dtype=np.int64)
