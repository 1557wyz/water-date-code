"""Library GA / NSGA-II with common genes, repair, evaluator and exact budget.

Reference implementation: pymoo 0.6.1.6. This module does not redefine NSGA-II.
All arms share SBX/PM and a 4n real-valued chromosome in [0,1]. RRP uses priority
and rotation genes; coordinate repair additionally uses the preferred x/y genes.
The inactive RRP coordinates are an explicit representation limitation.
"""
import time
from dataclasses import dataclass, asdict

import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems.static import StaticProblem

import water_plant_layout_v3 as model
from revision_coordinate_repair import repair_coordinates, coordinate_gene_scale
from revision_decoders import decode
from revision_evaluation import evaluate_layout, scalar_score
from revision_ga import dominates


@dataclass(frozen=True)
class BaselineConfig:
    algorithm: str
    decoder: str
    population: int
    evaluation_budget: int
    seed: int
    weights: tuple
    scales: tuple
    coordinate_scale_factor: float = 1.0
    initialization: str = 'random'

    def __post_init__(self):
        if self.algorithm not in ('ga', 'nsga2') or self.decoder not in ('coordinate_repair', 'rrp'):
            raise ValueError('invalid baseline arm')
        if type(self.population) is not int or self.population < 2:
            raise ValueError('integer population >=2 required')
        if type(self.evaluation_budget) is not int or self.evaluation_budget < self.population:
            raise ValueError('budget must cover initialization')
        if not np.isfinite(self.coordinate_scale_factor) or self.coordinate_scale_factor <= 0:
            raise ValueError('positive finite coordinate scale factor required')
        if self.initialization not in ('random', 'process_priority'):
            raise ValueError('unknown initialization')
        scalar_score({'feasible': False}, self.weights, self.scales)


def initial_vectors(facilities, settings):
    """Apply process ordering to 25% of initial individuals, using input links only.

    Rotations, preferred coordinates and remaining individuals are identical to
    random initialization. No historical optimized layout or outcomes are used.
    """
    n = len(facilities)
    initial = np.random.default_rng(settings.seed).random((settings.population, 4*n))
    if settings.initialization == 'process_priority':
        names = {f.name: i for i, f in enumerate(facilities)}
        process = []
        for system in ('main_water', 'advanced_water', 'sludge'):
            for pair in model.PIPELINE_SYSTEM[system]['pipes']:
                for name in pair:
                    if name in names and names[name] not in process:
                        process.append(names[name])
        for row in initial[:max(1, settings.population//4)]:
            remainder = [int(i) for i in np.argsort(row[:n], kind='stable') if i not in process]
            order = process+remainder
            priority_values = np.sort(row[:n]).copy()
            row[order] = priority_values
    return initial


def vector_to_layout(vector, facilities, decoder, coordinate_scale_factor=1.0):
    n = len(facilities)
    x = np.asarray(vector, dtype=float)
    if x.shape != (4*n,) or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
        raise ValueError('a finite 4n vector in [0,1] is required')
    chrom = {'order': np.argsort(x[:n], kind='stable').tolist(),
             'rotations': (x[n:2*n] >= .5).tolist()}
    if decoder == 'rrp':
        return decode(facilities, chrom, None, model.get_safety_distance)
    if decoder != 'coordinate_repair':
        raise ValueError('unknown decoder')
    scale = coordinate_gene_scale(facilities)*coordinate_scale_factor
    preferred = list(zip(x[2*n:3*n]*scale, x[3*n:]*scale))
    return repair_coordinates(facilities, chrom, preferred, model.get_safety_distance)


def run_baseline(facilities, evaluation_config, settings, progress_callback=None):
    if evaluation_config.bounds is not None:
        raise ValueError('these baselines implement the confirmed free-boundary case only')
    n = len(facilities)
    if n < 2:
        raise ValueError('at least two facilities required')
    started = time.perf_counter()
    # Same sampled 4n vectors for all algorithms/decoders for each seed.
    initial = initial_vectors(facilities, settings)
    options = dict(pop_size=settings.population, sampling=initial.copy(),
                   crossover=SBX(prob=.9, eta=15),
                   mutation=PM(prob=1.0, prob_var=1/(2*n), eta=20),
                   eliminate_duplicates=False)
    algorithm = GA(**options) if settings.algorithm == 'ga' else NSGA2(**options)
    if settings.algorithm == 'nsga2':
        algorithm.tournament_type = 'comp_by_rank_and_crowding'
    problem = Problem(n_var=4*n, n_obj=1 if settings.algorithm == 'ga' else 4,
                      n_ieq_constr=1, xl=0.0, xu=1.0)
    algorithm.setup(problem, termination=NoTermination(), seed=settings.seed, verbose=False)
    log, archive = [], []
    best = None
    while len(log) < settings.evaluation_budget:
        infills = algorithm.ask()
        if infills is None or len(infills) == 0:
            raise RuntimeError('algorithm returned no candidates before budget exhaustion')
        infills = infills[:settings.evaluation_budget-len(log)]
        rows, violations = [], []
        for vector in infills.get('X'):
            layout, _, _ = vector_to_layout(vector, facilities, settings.decoder, settings.coordinate_scale_factor)
            result = evaluate_layout(layout, facilities, evaluation_config)
            score = scalar_score(result, settings.weights, settings.scales)
            record = {'evaluation_id': len(log)+1, 'vector': vector.tolist(),
                      'score': float(score) if np.isfinite(score) else None,
                      'evaluation': result}
            log.append(record)
            if result['feasible']:
                obj = result['objectives']
                if best is None or score < best['score']:
                    best = record
                if not any(r['evaluation']['objectives'] == obj or dominates(r['evaluation']['objectives'], obj) for r in archive):
                    archive = [r for r in archive if not dominates(obj, r['evaluation']['objectives'])]
                    archive.append(record)
                rows.append([score] if settings.algorithm == 'ga' else
                            [v/s for v, s in zip(obj, settings.scales)])
                violations.append([0.0])
            else:
                # CV is explicitly binary, not a physical violation magnitude.
                # Feasibility-first prevents placeholders winning against feasible candidates.
                rows.append([0.0]*problem.n_obj)
                violations.append([1.0])
        static = StaticProblem(problem, F=np.asarray(rows), G=np.asarray(violations))
        algorithm.evaluator.eval(static, infills, skip_already_evaluated=False)
        algorithm.tell(infills=infills)
        if progress_callback is not None:
            progress_callback({'evaluation_count': len(log),
                               'best_score': best['score'] if best else None})
    if algorithm.evaluator.n_eval != len(log):
        raise RuntimeError('library and objective evaluation counts disagree')
    # Scores only choose a post-hoc representative for NSGA-II, never its mating/survival.
    return {'implementation': f'pymoo {pymoo.__version__}', 'settings': asdict(settings),
            'evaluation_count': len(log), 'library_evaluation_count': algorithm.evaluator.n_eval,
            'runtime_seconds': time.perf_counter()-started,
            'initial_vectors': initial.tolist(), 'best_by_reported_weights': best,
            'archive': archive, 'evaluations': log,
            'final_population_vectors': algorithm.pop.get('X').tolist(),
            'selection': 'scalar fitness' if settings.algorithm == 'ga' else 'rank and crowding',
            'operators': {'sbx_probability': .9, 'sbx_eta': 15, 'pm_probability': 1.0,
                          'pm_probability_per_variable': 1/(2*n), 'pm_eta': 20,
                          'duplicate_elimination': False}}
