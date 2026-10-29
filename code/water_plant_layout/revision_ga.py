"""Budgeted steady-state GA for revision experiments, separate from legacy code."""
import copy
import math
import random
import time
from dataclasses import dataclass

import water_plant_layout_v3 as model
from revision_decoders import decode
from revision_evaluation import scalar_score, evaluate_layout
from revision_rl import ActionController, RLConfig, mutate, state_of, reward_for


@dataclass(frozen=True)
class GAConfig:
    population: int
    evaluation_budget: int
    seed: int
    decoder: str
    controller: str
    weights: tuple
    scales: tuple

    def __post_init__(self):
        if type(self.population) is not int or self.population < 2:
            raise ValueError('population must be an integer >=2')
        if type(self.evaluation_budget) is not int or self.evaluation_budget < self.population:
            raise ValueError('budget must cover initialization')
        if self.decoder not in ('rrp', 'edge') or self.controller not in ('q_learning', 'random'):
            raise ValueError('invalid experimental arm')
        scalar_score({'feasible': False}, self.weights, self.scales)


def crossover(a, b, rng):
    """Order crossover and uniform rotation inheritance; consumes no evaluations."""
    n = len(a['order'])
    lo, hi = sorted(rng.sample(range(n), 2))
    order = [None]*n
    order[lo:hi+1] = a['order'][lo:hi+1]
    remaining = iter(x for x in b['order'] if x not in order)
    for i in range(n):
        if order[i] is None:
            order[i] = next(remaining)
    return {'order': order, 'rotations': [rng.choice((x, y)) for x, y in zip(a['rotations'], b['rotations'])]}


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def run_ga(facilities, evaluation_config, config):
    """Count every decode/evaluate request, including repeated chromosomes.

    Each mutation is a one-step episode starting at the evaluated crossover
    candidate. Terminal Q updates isolate immediate action effects; this is
    contextual action-value learning, not a claimed long-horizon GA MDP.
    """
    if len(facilities) < 2:
        raise ValueError('at least two facilities required')
    started = time.perf_counter()
    rng = random.Random(config.seed)
    agent = ActionController(config.controller, RLConfig(gamma=0), seed=config.seed)
    population, archive, evaluations, transitions = [], [], [], []
    best = None

    def evaluate(chrom, phase):
        nonlocal best, archive
        if len(evaluations) >= config.evaluation_budget:
            raise RuntimeError('evaluation budget exhausted')
        layout, _, _ = decode(facilities, chrom, evaluation_config.bounds, model.get_safety_distance, config.decoder)
        result = evaluate_layout(layout, facilities, evaluation_config)
        score = scalar_score(result, config.weights, config.scales)
        record = {'id': len(evaluations)+1, 'chromosome': copy.deepcopy(chrom),
                  'result': result, 'score': score}
        if result['feasible']:
            if best is None or score < best['score']:
                best = record
            obj = result['objectives']
            if not any(r['result']['objectives'] == obj or dominates(r['result']['objectives'], obj) for r in archive):
                archive = [r for r in archive if not dominates(obj, r['result']['objectives'])]
                archive.append(record)
        evaluations.append({'id': record['id'], 'phase': phase, 'feasible': result['feasible'],
                            'score': score if math.isfinite(score) else None,
                            'objectives': result['objectives'], 'errors': result['errors'],
                            'best_score': best['score'] if best else None})
        return record

    for _ in range(config.population):
        order = list(range(len(facilities)))
        rng.shuffle(order)
        chrom = {'order': order, 'rotations': [rng.choice((False, True)) for _ in facilities]}
        population.append(evaluate(chrom, 'initialization'))
    initial_chromosomes = [copy.deepcopy(r['chromosome']) for r in population]

    def select():
        return min(rng.sample(population, 2), key=lambda r: r['score'])

    while len(evaluations) < config.evaluation_budget:
        base_chrom = crossover(select()['chromosome'], select()['chromosome'], rng)
        base = evaluate(base_chrom, 'crossover')
        candidate = base
        if len(evaluations) < config.evaluation_budget:
            state = state_of(base['result'])
            action = agent.select(state)
            child = evaluate(mutate(base_chrom, action, agent.mutation_rng), 'mutation')
            reward = reward_for(base['result'], child['result'], base['score'], child['score'])
            candidate = child if child['score'] < base['score'] else base
            before, after = agent.update(state, action, reward, state_of(candidate['result']), terminal=True)
            transitions.append({'base_evaluation': base['id'], 'child_evaluation': child['id'],
                                'state': state, 'action': action, 'reward': reward,
                                'selected_evaluation': candidate['id'], 'q_before': before, 'q_after': after,
                                'terminal': True})
        worst = max(range(len(population)), key=lambda i: population[i]['score'])
        if candidate['score'] < population[worst]['score']:
            population[worst] = candidate

    def export(record):
        if record is None:
            return None
        return {'evaluation_id': record['id'], 'chromosome': record['chromosome'],
                'score': record['score'], 'evaluation': record['result']}
    return {'evaluation_count': len(evaluations), 'unique_chromosomes_in_final_population':
                len({(tuple(r['chromosome']['order']), tuple(r['chromosome']['rotations'])) for r in population}),
            'runtime_seconds': time.perf_counter()-started,
            'initial_chromosomes': initial_chromosomes, 'best': export(best),
            'archive': [export(r) for r in archive], 'evaluations': evaluations,
            'transitions': transitions,
            'nonzero_q_entries': sum(v != 0 for row in agent.q.values() for v in row.values()),
            'rl_config': {'alpha': .1, 'epsilon': .2, 'gamma': 0, 'episode': 'one mutation'}}
