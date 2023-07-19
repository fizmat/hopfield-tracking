import argparse
import logging
from typing import Dict

import pandas as pd
import seaborn as sns
from ConfigSpace import ConfigurationSpace, Configuration
from matplotlib import pyplot as plt
from smac import MultiFidelityFacade, Scenario

from datasets import get_hits, get_datasets
from hopfield import iterate
from hopfield.iterate import metric_history
from metrics.tracks import track_metrics
from segment.track import gen_seg_track_layered

logging.basicConfig(level=logging.WARNING)


class Optimizer:
    def __init__(self, dataset: str, n_events: int = None, event_size: int = None, n_noise_hits=100):
        self.hits = get_hits(dataset, n_events=n_events, n_noise_hits=n_noise_hits, event_size=event_size)
        self.events = {str(eid): event.reset_index(drop=True) for eid, event in self.hits.groupby('event_id')}

    def evaluate(self, config: Configuration, instance: str, seed: int = 0) -> Dict:
        event = self.events[instance]
        seg, acts, positives = iterate.run(event, **config)
        tseg = gen_seg_track_layered(event)
        score = track_metrics(event, seg, tseg, acts[-1], positives[-1])
        score['total steps'] = config['cooling_steps'] + config['rest_steps']
        score['trackml loss'] = 1. - score['trackml score']
        return score['trackml loss']

    def get_configspace(self):
        return ConfigurationSpace({
            'alpha': (0., 20.),
            'gamma': (0., 20.),
            'bias': (-10.0, 10.0),
            'threshold': 0.5,
            'cosine_power': (0.0, 20.0),
            'cosine_min_allowed': (-1., 1.),
            'cosine_min_rewarded': (0., 1.),
            'distance_op': ['sum', 'prod'],
            'distance_power': (0., 3.),
            't_min': 1.,
            't_max': (1., 100.),
            'cooling_steps': (1, 100),
            'rest_steps': (1, 50),
            'initial_act': (0., 1.),
            'dropout': 0,
            'learning_rate': (0., 1.)
        })

    def run(self, args, n_jobs=-1):
        scenario = Scenario(
            self.get_configspace(),
            n_trials=args.runcount_limit,
            instances=list(self.events.keys()),
            instance_features={k: [len(v)] for k, v in self.events.items()},
            n_workers=n_jobs
        )

        optimizer = MultiFidelityFacade(scenario, self.evaluate)
        best_config = optimizer.optimize()
        vtrain_history = optimizer.validate(best_config)
        return best_config, optimizer.runhistory, optimizer.intensifier.trajectory, vtrain_history


def main():
    parser = argparse.ArgumentParser(description='Optimize hopfield-tracking')
    parser.add_argument('--dataset', type=str, default='spdsim', help='Dataset identifier string',
                        choices=get_datasets())
    parser.add_argument('--n-jobs', type=int, default=-1, help='smac optimization parallelism')
    parser.add_argument('--n-events', type=int, default=100,
                        help='Total number of events to read/generate for training.')
    parser.add_argument('--max-event-size', type=int, default=10, help='Max event size (in tracks).')
    parser.add_argument('--example-event-size', type=int, default=10,
                        help='Event size (in tracks) to visualize one sample event.')
    parser.add_argument('--runcount-limit', type=int, default=10,
                        help='Maximum number of runs to perform')

    args = parser.parse_args()

    optimizer = Optimizer(args.dataset, args.n_events, args.max_event_size)
    best_config, history, trajectory, vtrain_history = optimizer.run(args)
    print(best_config)
    event = [event for eid, event in optimizer.hits.groupby('event_id')
             if event[event.track >= 0].track.nunique() == args.example_event_size][0].reset_index(drop=True)
    seg, acts, positives = iterate.run(event, **best_config)
    tseg = gen_seg_track_layered(event)
    metrics = metric_history(event, seg, tseg, acts, positives)
    sns.relplot(data=metrics.reset_index().melt('index'), row='variable', x='index', y='value', kind="line",
                facet_kws={'sharey': False, 'sharex': True})
    plt.savefig('metrics.png')
    df = pd.DataFrame(trajectory)
    assert (df.config_ids.str.len() == 1).all()
    df.config_ids = df.config_ids.str[0]
    assert (df.costs.str.len() == 1).all()
    df.costs = df.costs.str[0]
    sns.relplot(data=df.drop(columns=['config_ids', 'costs']).reset_index().melt('index'), row='variable', x='index',
                y='value', kind="line",
                facet_kws={'sharey': False, 'sharex': True})
    plt.savefig('trajectory.png')
    print(vtrain_history)


if __name__ == '__main__':
    main()
