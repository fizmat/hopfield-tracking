from functools import partial

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from smac import MultiFidelityFacade, Scenario

from datasets import get_hits
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import curvature_energy_matrix, segment_adjacent_pairs
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


def main():
    configspace = ConfigurationSpace({
        'alpha': (0., 20.),
        'gamma': (0., 20.),
        'bias': (-10.0, 10.0),
        'cosine_power': (0.0, 20.0),
        'cosine_min_allowed': (-1., 1.),
        'cosine_min_rewarded': (0., 1.),
        'distance_op': ['sum', 'prod'],
        'distance_power': (0., 3.),
        't_max': (1., 100.),
        'cooling_steps': (1, 100),
        'rest_steps': (1, 50),
        'initial_act': (0., 1.),
        'learning_rate': (0., 1.)
    })

    N_EVENTS = 3
    scenario = Scenario(
        configspace,
        n_trials=10,
        n_workers=1,
        min_budget=1,
        max_budget=N_EVENTS
    )
    events = get_hits('spdsim', n_events=N_EVENTS, n_noise_hits=1000, event_size=10)

    events = {eid: event.reset_index(drop=True) for eid, event in events.groupby('event_id')}
    eids = tuple(events.keys())

    def evaluate(config: Configuration, seed: int, budget: float = 10) -> float:
        conf = CONFIG_DEFAULTS.copy()
        conf.update(config)
        rng = np.random.default_rng(seed=seed)
        scores = []
        for eid in rng.choice(eids, int(budget), replace=False):
            event = events[eid]
            seg = gen_seg_layered(event)
            crossing_matrix = conf['alpha'] * cross_energy_matrix(seg)
            curvature_matrix = curvature_energy_matrix(
                pos=event[['x', 'y', 'z']].to_numpy(),
                seg=seg, pairs=segment_adjacent_pairs(seg), alpha=conf['alpha'], gamma=conf['gamma'],
                cosine_threshold=conf['cosine_min_rewarded'], cosine_min_allowed=conf['cosine_min_allowed'],
                curvature_cosine_power=conf['cosine_power'],
                do_sum_r=conf['distance_op'] == 'sum', distance_prod_power_in_denominator=conf['distance_power']
            )
            energy_matrix = crossing_matrix + curvature_matrix
            temp_curve = annealing_curve(conf['t_min'], conf['t_max'],
                                         conf['cooling_steps'], conf['rest_steps'])
            act = np.full(len(seg), conf['initial_act'])
            update_act = partial(update_act_bulk, learning_rate=conf['learning_rate'])
            for _ in anneal(energy_matrix, temp_curve, act, update_act, bias=conf['bias']):
                pass
            tseg = gen_seg_track_layered(event)
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
