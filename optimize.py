from functools import partial

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from smac import Scenario, MultiFidelityFacade

from datasets import get_hits, bman
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import curvature_energy_matrix, kink_energy_matrix, \
    prep_curvature
from hopfield.iterate import annealing_curve, update_act_bulk, anneal
from metrics.tracks import track_metrics
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_layered

CONFIG_DEFAULTS = {
    'alpha': 1.,
    'gamma': 1.,
    'bias': 0.,
    'cosine_power': 1.,
    'cosine_min_allowed': -1.1,
    'cosine_min_rewarded': 0,
    'distance_op': 'sum',
    'distance_power': 1.,
    'threshold': 0.5,
    't_min': 1.,
    't_max': 1.,
    'cooling_steps': 10,
    'rest_steps': 10,
    'initial_act': .5,
    'learning_rate': 1.
}


def preprocess(g):
    event = g.reset_index(drop=True)
    seg = gen_seg_layered(event)
    tseg = gen_seg_track_layered(event)
    pairs, cosines, r1, r2 = prep_curvature(event[['x', 'y', 'z']].to_numpy(), seg)
    return {
        'event': event,
        'seg': seg,
        'tseg': tseg,
        'pairs': pairs,
        'cosines': cosines,
        'r1': r1,
        'r2': r2,
        'cross_matrix': cross_energy_matrix(seg),
    }


def main():
    N_EVENTS = 50
    N_TRIALS = 200

    hits = get_hits('bman', n_events=N_EVENTS)
    hits[['x', 'y', 'z']] /= bman.LAYER_DIST
    hits = hits[hits['track'] != -1]
    geometry = hits.groupby('event_id').apply(preprocess)

    extra_conf = {
        'cooling_steps': 50,
        'rest_steps': 5,
    }

    scenario = Scenario(
        ConfigurationSpace({
            'alpha': (0., 1000.),
            'gamma': (0., 2000.),
            'bias': (-200.0, 200.0),
            'cosine_power': (0.0, 50.0),
            'cosine_min_rewarded': (0., 1.),
            't_max': (1., 1000.),
            'initial_act': (0., 1.),
        }),
        'bulk-norate',
        n_trials=N_TRIALS,
        min_budget=1,
        max_budget=N_EVENTS
    )

    def evaluate(config: Configuration, seed: int, budget: float) -> float:
        conf = CONFIG_DEFAULTS.copy()
        conf.update(extra_conf)
        conf.update(config)
        rng = np.random.default_rng(seed=seed)
        scores = []
        for eid in rng.choice(geometry.index, int(budget), replace=False):
            event, seg, tseg, pairs, cosines, r1, r2, crossing_matrix = geometry[eid].values()
            curvature_matrix = curvature_energy_matrix(
                len(seg), pairs, cosines, r1, r2,
                cosine_power=conf['cosine_power'], cosine_threshold=conf['cosine_min_rewarded'],
                distance_power=conf['distance_power']
            )
            kink_matrix = kink_energy_matrix(len(seg), pairs, cosines, conf['cosine_min_allowed'])
            energy_matrix = conf['alpha'] * (crossing_matrix + kink_matrix) - conf['gamma'] * curvature_matrix
            temp_curve = annealing_curve(conf['t_min'], conf['t_max'],
                                         conf['cooling_steps'], conf['rest_steps'])
            act = np.full(len(seg), conf['initial_act'])
            update_act = partial(update_act_bulk, learning_rate=conf['learning_rate'])
            for _ in anneal(energy_matrix, temp_curve, act, update_act, bias=conf['bias']):
                pass

            positive = act > conf['threshold']
            score = track_metrics(event, seg, tseg, act, positive)
            score['total steps'] = conf['cooling_steps'] + conf['rest_steps']
            score['trackml loss'] = 1. - score['trackml score']
            scores.append(score)
        return pd.DataFrame(scores).mean()['trackml loss']

    optimizer = MultiFidelityFacade(scenario, evaluate, overwrite=True)
    best_config = optimizer.optimize()
    print(dict(best_config))
    print(pd.DataFrame(optimizer.intensifier.trajectory))
    print(optimizer.validate(best_config))


if __name__ == '__main__':
    main()
