import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from smac import MultiFidelityFacade, Scenario

from datasets import get_hits
from hopfield import iterate
from metrics.tracks import track_metrics
from segment.track import gen_seg_track_layered


def main():
    configspace = ConfigurationSpace({
        'alpha': (0., 20.),
        'gamma': (0., 20.),
        'bias': (-10.0, 10.0),
        # 'threshold': 0.5,
        'cosine_power': (0.0, 20.0),
        'cosine_min_allowed': (-1., 1.),
        'cosine_min_rewarded': (0., 1.),
        'distance_op': ['sum', 'prod'],
        'distance_power': (0., 3.),
        # 't_min': 1.,
        't_max': (1., 100.),
        'cooling_steps': (1, 100),
        'rest_steps': (1, 50),
        'initial_act': (0., 1.),
        'learning_rate': (0., 1.)
    })

    N_EVENTS = 20
    scenario = Scenario(
        configspace,
        n_trials=200,
        n_workers=-1,
        min_budget=1,
        max_budget=N_EVENTS
    )
    events = get_hits('spdsim', n_events=N_EVENTS, n_noise_hits=100, event_size=10)

    events = {eid: event.reset_index(drop=True) for eid, event in events.groupby('event_id')}
    eids = tuple(events.keys())

    def evaluate(config: Configuration, seed: int, budget: float = 10) -> float:
        rng = np.random.default_rng(seed=seed)
        scores = []
        for eid in rng.choice(eids, int(budget), replace=False):
            event = events[eid]
            seg, acts, positives = iterate.run(event, **config)
            tseg = gen_seg_track_layered(event)
            score = track_metrics(event, seg, tseg, acts[-1], positives[-1])
            score['total steps'] = config['cooling_steps'] + config['rest_steps']
            score['trackml loss'] = 1. - score['trackml score']
            scores.append(score)
        return pd.DataFrame(scores).mean()['trackml loss']

    optimizer = MultiFidelityFacade(scenario, evaluate)
    best_config = optimizer.optimize()
    print(dict(best_config))
    print(pd.DataFrame(optimizer.intensifier.trajectory))
    print(optimizer.validate(best_config))


if __name__ == '__main__':
    main()
