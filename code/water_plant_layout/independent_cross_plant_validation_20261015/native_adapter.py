"""Bind instance data to an isolated original module; observe, never change, search."""
from __future__ import annotations
import ast
import copy
import importlib.util
import math
import threading
import time

import numpy as np
from common import CORE, digest


class SerialEvaluationMap:
    """Same ordered map contract, without nested worker pools in each run."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, values):
        return map(function, values)


def load_core(serial=True):
    spec = importlib.util.spec_from_file_location("independent_original_water_model", CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if serial:
        module.ThreadPoolExecutor = SerialEvaluationMap
    return module


def native_named_pairs():
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_calc_process_adjacency_strict")
    result = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            if all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in node.elts):
                result.add(tuple(v.value for v in node.elts))
    return result


def configure(module, instance, variant):
    constraints = instance["constraint_versions"][variant]
    zone_map = constraints.get("category_overrides", {})
    facilities = [module.Facility(f["id"], f["label"], f["length_m"], f["width_m"],
        zone_map.get(f["id"], f["category"]), f["safety_level"]) for f in instance["facilities"]]
    module.FACILITIES_DATA = [(f.name, f.name_cn, f.length, f.width, f.category, f.safety_level) for f in facilities]
    module.SAFETY_DISTANCES = {tuple(row["levels"]): row["distance_m"] for row in constraints["safety_matrix"]}
    module.ADJACENCY_REQUIRED = [(e["source"], e["target"], e["adjacency_weight"]) for e in instance["contacts"]]
    module.PIPELINE_SYSTEM = {}
    for edge in instance["contacts"]:
        module.PIPELINE_SYSTEM.setdefault(edge["system"], {"pipes": [], "label": edge["system"]})["pipes"].append((edge["source"], edge["target"]))
    module.PIPE_SEGMENT_TO_SYSTEM = {(e["source"], e["target"]): e["system"] for e in instance["contacts"]}
    module.ELEVATION_DATA = instance["elevations"]
    module.ELEV_SEGMENT_PARAMS = {(r["source"], r["target"]): tuple(r["parameters"])
                                   for r in instance["hydraulic_segments"]}
    module.MAIN_WATER_ELEV_PATH = [(r["source"], r["target"]) for r in instance["hydraulic_segments"]]
    module.SLUDGE_ELEV_PATH = []
    return facilities


def serialize_layout(layout):
    return [{"id": f.name, "x_m": float(f.x), "y_m": float(f.y),
        "length_m": float(f.get_dimensions()[0]), "width_m": float(f.get_dimensions()[1]),
        "rotated": bool(f.rotated), "category": f.category, "safety_level": f.safety_level} for f in layout]


def audit_layout(module, layout, instance, variant):
    expected = {f["id"] for f in instance["facilities"]}
    names = [f.name for f in layout]
    if not layout:
        return {"spatial_feasible": False, "hydraulic_feasible": False, "evaluated_hydraulic_segments": 0,
                "missing_count": len(expected), "duplicate_count": 0}
    rectangles = np.array([(f.x, f.y, *f.get_dimensions()) for f in layout], dtype=float)
    x, y, w, h = rectangles.T
    i, j = np.triu_indices(len(layout), 1)
    dx = np.maximum(0, np.maximum(x[i], x[j])-np.minimum(x[i]+w[i], x[j]+w[j]))
    dy = np.maximum(0, np.maximum(y[i], y[j])-np.minimum(y[i]+h[i], y[j]+h[j]))
    overlaps = (x[i] < x[j]+w[j]-1e-7) & (x[j] < x[i]+w[i]-1e-7) & (y[i] < y[j]+h[j]-1e-7) & (y[j] < y[i]+h[i]-1e-7)
    gaps = np.hypot(dx, dy)
    required = np.array([module.get_safety_distance(layout[a], layout[b]) for a, b in zip(i, j)])
    shortage = np.maximum(0, required-gaps)
    missing = len(expected-set(names))
    duplicate = len(names)-len(set(names))
    invalid = not np.isfinite(rectangles).all() or bool(np.any(w <= 0) or np.any(h <= 0))
    negative = int(np.count_nonzero((x < -1e-7) | (y < -1e-7)))
    bw, bh = float(np.max(x+w)), float(np.max(y+h))
    spatial = not (missing or duplicate or invalid or negative or np.any(overlaps) or np.any(shortage > 1e-6))
    fdict = {f.name: f for f in layout}
    adjacency = []
    for e in instance["contacts"]:
        if e["source"] not in fdict or e["target"] not in fdict:
            adjacency.append(float("inf"))
            continue
        a, b = fdict[e["source"]], fdict[e["target"]]
        gx, gy = module.facility_edge_gap(a, b)
        adjacency.append(max(0, math.hypot(gx, gy)-e["target_gap_m"]))
    hydraulic = []
    for r in instance["hydraulic_segments"]:
        up, down = r["source"], r["target"]
        if up not in fdict or down not in fdict:
            continue
        a, b = instance["elevations"][up], instance["elevations"][down]
        length = module.calculate_pipeline_segment(fdict[up], fdict[down], module.infer_pipeline_system(up, down))["routed_length_m"]
        slope, local, margin, weight = r["parameters"]
        downstream = b["ground"] + (-0.5 if b["type"] == "pump" else 0.3)
        source = module.SB_SLUDGE_OUTLET_ELEV if (up, down) == ("SB", "STK") else a["water_out"]
        hydraulic.append(max(0, slope*length+local+margin-(source-downstream)))
    hydraulic_ok = len(hydraulic) == len(instance["hydraulic_segments"]) and all(v <= 1e-7 for v in hydraulic)
    site = instance["reference_site"]
    site_fit = bw <= site["width_m"]+1e-6 and bh <= site["height_m"]+1e-6
    zones = {}
    for f in layout:
        zones.setdefault(f.category, []).append(f)
    zone_fill = []
    for group in zones.values():
        if len(group) > 1:
            area = (max(f.x+f.get_dimensions()[0] for f in group)-min(f.x for f in group))*(max(f.y+f.get_dimensions()[1] for f in group)-min(f.y for f in group))
            zone_fill.append(sum(f.area for f in group)/area if area > 0 else 0)
    return {"spatial_feasible": bool(spatial), "hydraulic_feasible": bool(hydraulic_ok),
        "combined_model_feasible": bool(spatial and hydraulic_ok),
        "missing_count": missing, "duplicate_count": duplicate, "overlap_pairs": int(np.sum(overlaps)),
        "safety_violation_pairs": int(np.sum(shortage > 1e-6)), "safety_shortage_sum_m": float(np.sum(shortage)),
        "negative_coordinate_count": negative, "bbox_width_m": bw, "bbox_height_m": bh, "bbox_area_m2": bw*bh,
        "footprint_utilization": sum(f.area for f in layout)/(bw*bh) if bw*bh > 0 else 0,
        "adjacency_violation_contacts": sum(v > 1e-6 for v in adjacency),
        "adjacency_satisfaction": sum(v <= 1e-6 for v in adjacency)/len(adjacency),
        "adjacency_excess_sum_m": sum(v for v in adjacency if math.isfinite(v)),
        "hydraulic_violation_segments": sum(v > 1e-7 for v in hydraulic),
        "hydraulic_shortage_sum_m": sum(hydraulic), "evaluated_hydraulic_segments": len(hydraulic),
        "reference_site_fit_posthoc_only": bool(site_fit),
        "mean_zone_fill": float(np.mean(zone_fill)) if zone_fill else None,
        "low_fill_zones_below_0_4": sum(v < 0.4 for v in zone_fill)}


def components(ga, layout, bw, bh):
    area = bw*bh
    util = ga.total_area/area if area > 0 else 0
    bonus = -60000*util
    if util > .72: bonus -= 40000
    elif util > .70: bonus -= 32000
    elif util > .68: bonus -= 25000
    elif util > .65: bonus -= 18000
    elif util > .62: bonus -= 12000
    elif util > .58: bonus -= 6000
    elif util > .55: bonus -= 3000
    elif util < .50: bonus += 10000
    aspect = max(bw, bh)/min(bw, bh) if min(bw, bh) > 0 else 2
    terms = {"process": ga._calc_process_adjacency_strict(layout),
        "area": area*6.5+(area/1000)**1.5*100, "utilization": bonus,
        "aspect": (2-aspect)*50 if aspect < 2 else ((aspect-3)*120 if aspect > 3 else 0),
        "pipeline": .7*ga._calc_pipeline_length(layout), "zone_compactness": 1.2*ga._calc_zone_compactness(layout),
        "dormitory": .7*ga._calc_dormitory_cluster_penalty(layout),
        "zone_boundary": .5*ga._calc_zone_boundary_clarity(layout), "alignment": .4*ga._calc_alignment_bonus(layout),
        "boundary": ga._calc_boundary_tightness(layout, bw, bh),
        "elevation": 400*ga._calc_elevation_constraint(layout)}
    terms = {key: float(value) for key, value in terms.items()}
    return {"terms": terms, "score": math.fsum(terms.values()),
        "weighted_pipeline_m": ga._calc_pipeline_length(layout)}


def observed_class(module, instance, variant, stream):
    class ObservedGA(module.AdvancedGA):
        def __init__(self, facilities, **kwargs):
            super().__init__(facilities, **kwargs)
            self.observation_lock = threading.RLock()
            self.fitness_calls = 0
            self.unique_evaluations = 0
            self.audit_count = 0
            self.best_spatial = None
            self.best_combined = None
            self.last_observed = None
            self.started = time.perf_counter()

        def calculate_fitness(self, chrom):
            with self.observation_lock:
                self.fitness_calls += 1
                repaired = self._repair_chromosome(chrom)
                key = self._chromosome_key(repaired)
                cached = self.enable_cache and key in self._fitness_cache
                before = self.audit_count
                value = super().calculate_fitness(chrom)
                if not cached:
                    self.unique_evaluations += 1
                    if self.audit_count > before and self.last_observed is not None:
                        layout, bw, bh, audit = self.last_observed
                        row = {"evaluation": self.unique_evaluations, "fitness": float(value) if math.isfinite(value) else None,
                            "elapsed_s": time.perf_counter()-self.started, "chromosome": repaired,
                            "audit": audit, "layout": serialize_layout(layout)}
                        stream.write(__import__("json").dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                        for flag, attr in (("spatial_feasible", "best_spatial"), ("combined_model_feasible", "best_combined")):
                            old = getattr(self, attr)
                            if audit.get(flag) and math.isfinite(value) and (old is None or value < old["score"]):
                                setattr(self, attr, {"score": float(value), "layout": serialize_layout(layout),
                                    "audit": audit, "chromosome": copy.deepcopy(repaired),
                                    "components": components(self, layout, bw, bh), "evaluation": self.unique_evaluations})
                return value

        def decode_chromosome(self, chrom, container_size):
            result = super().decode_chromosome(chrom, container_size)
            with self.observation_lock:
                self.audit_count += 1
                self.last_observed = (*result, audit_layout(module, result[0], instance, variant))
            return result
    return ObservedGA
