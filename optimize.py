import argparse
import logging
from argparse import ArgumentError
from typing import Dict

import pandas as pd
import seaborn as sns
from ConfigSpace import ConfigurationSpace, Configuration
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from vispy import app

from datasets import get_hits, get_datasets
from hopfield import iterate
from hopfield.iterate import metric_history
from hopfield.plot import plot_result
from metrics.segments import gen_perfect_act
from metrics.tracks import track_metrics
from segment.track import gen_seg_track_layered

logging.basicConfig(level=logging.WARNING)


class Optimizer:
    def __init__(self, dataset: str, n_events: int = None, event_size: int = None, n_noise_hits=100):
        self.hits = get_hits(dataset, n_events=n_events, n_noise_hits=n_noise_hits, event_size=event_size)
        self.events = {str(eid): event.reset_index(drop=True) for eid, event in self.hits.groupby('event_id')}

    def evaluate(self, config: Configuration, instance: str) -> Dict:
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
        train, test = train_test_split(list(self.events.keys()))
        scenario = Scenario({
            'run_obj': 'quality',
            'runcount-limit': args.runcount_limit,
            'cs': self.get_configspace(),
            'instances': [[s] for s in train],
            'test_instances': [[s] for s in test]
        })

        if args.output_directory:
            scenario.output_dir = args.output_directory
            scenario.input_psmac_dirs = args.output_directory
        scenario.shared_model = args.batch
        optimizer = SMAC4MF(scenario=scenario, tae_runner=self.evaluate, n_jobs=n_jobs)
        best_config = optimizer.optimize()
        vtrain_history = optimizer.validate(instance_mode='train')
        vtest_history = optimizer.validate(instance_mode='test')
        return best_config, optimizer.get_runhistory(), optimizer.get_trajectory(), vtrain_history, vtest_history


def main():
    parser = argparse.ArgumentParser(description='Optimize hopfield-tracking')
    parser.add_argument('--dataset', type=str, default='spdsim', help='Dataset identifier string',
                        choices=get_datasets())
    parser.add_argument('--batch', action='store_true', help='Share output directory')
    parser.add_argument('--n-events', type=int, default=100,
                        help='Total number of events to read/generate for training.')
    parser.add_argument('--max-event-size', type=int, default=10, help='Max event size (in tracks).')
    parser.add_argument('--example-event-size', type=int, default=10,
                        help='Event size (in tracks) to visualize one sample event.')
    parser.add_argument('--runcount-limit', type=int, default=10,
                        help='Maximum number of runs to perform')
    parser.add_argument('--output-directory', type=str, default=None,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    args = parser.parse_args()

    if args.batch:
        raise ArgumentError(args.shared_directory,
                            'Shared output directory is required when running in a batch system')
    optimizer = Optimizer(args.dataset, args.n_events, args.max_event_size)
    best_config, history, trajectory, vtrain_history, vtest_history = optimizer.run(args)
    print(best_config.get_dictionary())
    event = [event for eid, event in optimizer.hits.groupby('event_id')
             if event[event.track >= 0].track.nunique() == args.example_event_size][0].reset_index(drop=True)
    seg, acts, positives = iterate.run(event, **best_config)
    tseg = gen_seg_track_layered(event)
    perfect_act = gen_perfect_act(seg, tseg)
    act = acts[-1]
    metrics = metric_history(event, seg, tseg, acts, positives)
    sns.relplot(data=metrics.reset_index().melt('index'), row='variable', x='index', y='value', kind="line",
                facet_kws={'sharey': False, 'sharex': True})
    plt.savefig('metrics.png')
    df = pd.DataFrame(trajectory).drop(columns='incumbent')
    sns.relplot(data=df.reset_index().melt('index'), row='variable', x='index', y='value', kind="line",
                facet_kws={'sharey': False, 'sharex': True})
    plt.savefig('trajectory.png')
    df = pd.DataFrame([rec for rec in pd.DataFrame(trajectory).incumbent])
    sns.relplot(data=df.reset_index().melt('index'), row='variable', x='index', y='value', kind="line",
                facet_kws={'sharey': False, 'sharex': True})
    plt.savefig('incumbent.png')
    df_val = pd.DataFrame([{'kind': 'train',
                            'event': int(k.instance_id),
                            'trackml_cost': v.cost
                            }
                           for k, v in vtrain_history.items()] +
                          [{'kind': 'test',
                            'event': int(k.instance_id),
                            'trackml_cost': v.cost
                            }
                           for k, v in vtest_history.items()]
                          ).set_index('event').sort_index()
    sns.stripplot(data=df_val, y='trackml_cost', x='kind')
    plt.savefig('iterate_trackml_cost.png')


if __name__ == '__main__':
    main()
