"""Auditable action-level Q-learning and matched random-action control.

This experimental controller uses the revised evaluator, never legacy fitness.
Its aggregated state is an approximate search observation, not a proven MDP.
"""
import copy
import math
import random
from dataclasses import dataclass

ACTIONS = ('swap', 'insert', 'reverse', 'rotate')


def mutate(chrom, action, rng):
    if action not in ACTIONS:
        raise ValueError('unknown action')
    child = copy.deepcopy(chrom)
    n = len(child['order'])
    if n < 2 or sorted(child['order']) != list(range(n)) or len(child['rotations']) != n:
        raise ValueError('valid order and rotations for at least two facilities required')
    if action == 'rotate':
        i = rng.randrange(n)
        child['rotations'][i] = not child['rotations'][i]
    else:
        i, j = rng.sample(range(n), 2)
        if action == 'swap':
            child['order'][i], child['order'][j] = child['order'][j], child['order'][i]
        elif action == 'insert':
            child['order'].insert(j, child['order'].pop(i))
        else:
            i, j = sorted((i, j))
            child['order'][i:j+1] = reversed(child['order'][i:j+1])
    return child


def state_of(result):
    if result['metrics'] is None:
        return ('invalid_geometry',)
    m = result['metrics']
    def level(x, a, b):
        return 0 if x < a else (1 if x < b else 2)
    return (bool(result['feasible']), level(m['utilization'], .50, .65),
            level(m['adjacency_satisfaction'], .50, .80),
            m['hydraulic_violating_segments'] > 0)


def reward_for(parent_result, child_result, parent_score, child_score):
    pf, cf = parent_result['feasible'], child_result['feasible']
    if pf != cf:
        return 1.0 if cf else -1.0
    if not pf:
        return 0.0
    if not math.isfinite(parent_score) or not math.isfinite(child_score):
        raise ValueError('feasible candidates require finite scores')
    return max(-1.0, min(1.0, (parent_score-child_score)/max(abs(parent_score), abs(child_score), 1.0)))


@dataclass(frozen=True)
class RLConfig:
    alpha: float = .1
    gamma: float = .9
    epsilon: float = .2

    def __post_init__(self):
        for name in ('alpha', 'gamma', 'epsilon'):
            v = getattr(self, name)
            if not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if self.alpha == 0:
            raise ValueError('alpha must be positive')


class ActionController:
    def __init__(self, mode, config=RLConfig(), seed=0):
        if mode not in ('q_learning', 'random'):
            raise ValueError('unknown controller mode')
        self.mode, self.config = mode, config
        self.selection_rng = random.Random(seed)
        self.mutation_rng = random.Random(seed+1000003)
        self.q = {}
        self.trace = []

    def select(self, state):
        if self.mode == 'random':
            return self.selection_rng.choice(ACTIONS)
        values = self.q.setdefault(state, {a: 0.0 for a in ACTIONS})
        if self.selection_rng.random() < self.config.epsilon:
            return self.selection_rng.choice(ACTIONS)
        best = max(values.values())
        return self.selection_rng.choice([a for a in ACTIONS if values[a] == best])

    def update(self, state, action, reward, next_state, terminal=False):
        if action not in ACTIONS or not math.isfinite(reward):
            raise ValueError('invalid transition')
        if self.mode == 'random':
            return None, None
        values = self.q.setdefault(state, {a: 0.0 for a in ACTIONS})
        future = 0.0 if terminal else max(self.q.setdefault(next_state, {a: 0.0 for a in ACTIONS}).values())
        before = values[action]
        values[action] += self.config.alpha*(reward+self.config.gamma*future-before)
        return before, values[action]

    def step(self, parent, evaluate, score, terminal=False):
        """Exactly two evaluation requests; accept only a strictly better child.

        `evaluate` and `score` are shared by both controllers. A future GA can
        reuse parent evaluations, but must count calls consistently across arms.
        Next state reflects the accepted candidate, including rejected actions.
        """
        pr = evaluate(parent)
        ps = score(pr)
        state = state_of(pr)
        action = self.select(state)
        child = mutate(parent, action, self.mutation_rng)
        cr = evaluate(child)
        cs = score(cr)
        reward = reward_for(pr, cr, ps, cs)
        accepted = bool(cr['feasible'] and (not pr['feasible'] or cs < ps))
        next_state = state_of(cr if accepted else pr)
        before, after = self.update(state, action, reward, next_state, terminal)
        # Non-finite scores are serialized as null, never nonstandard JSON Infinity.
        self.trace.append({'state': state, 'action': action, 'next_state': next_state,
                           'reward': reward, 'parent_score': ps if math.isfinite(ps) else None,
                           'child_score': cs if math.isfinite(cs) else None,
                           'parent_feasible': pr['feasible'], 'child_feasible': cr['feasible'],
                           'accepted': accepted, 'terminal': terminal,
                           'genotype_changed': child != parent, 'q_before': before, 'q_after': after,
                           'evaluation_requests': 2})
        return copy.deepcopy(child if accepted else parent)
