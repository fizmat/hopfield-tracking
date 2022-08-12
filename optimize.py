import argparse
import logging
from argparse import ArgumentError
from typing import Dict

import ConfigSpace as CS
import pandas as pd
import seaborn as sns
from ConfigSpace import Configuration
from matplotlib import pyplot as plt
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
    def __init__(self, dataset: str, n_events: int = None):
        self.hits = get_hits(dataset, n_events=n_events)

    def evaluate(self, config: Configuration, instance: str) -> Dict:
        event = self.hits[self.hits.event_id == int(instance)]
        event.reset_index(drop=True, inplace=True)
        seg, acts, positives = iterate.run(event, **config)
        tseg = gen_seg_track_layered(event)
        score = track_metrics(event, seg, tseg, acts[-1], positives[-1])
        score['total_steps'] = config['cooling_steps'] + config['rest_steps']
        score['trackml_loss'] = 1. - score['trackml']
        return score

    def get_configspace(self):
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('alpha', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('gamma', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bias', lower=-10, upper=10))
        config_space.add_hyperparameter(CS.Constant('threshold', value=0.5))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_power', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_min_allowed', lower=-1, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_min_rewarded', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('distance_power', lower=0, upper=3))
        config_space.add_hyperparameter(CS.Constant('t_min', value=1.))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('t_max', lower=1, upper=100))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('cooling_steps', lower=1, upper=100))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('rest_steps', lower=0, upper=50))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('initial_act', lower=0, upper=1))
        config_space.add_hyperparameter(CS.Constant('dropout', value=0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0, upper=1))
        return config_space

    def run(self, args):
        scenario = Scenario({
            'run_obj': 'quality',
            'runcount-limit': args.runcount_limit,
            'cs': self.get_configspace(),
            'instances': [[str(eid)] for eid in self.hits.event_id.unique()],
            'multi_objectives': ['trackml_loss', 'total_steps', 'n_fp_seg']
        })

        if args.output_directory:
            scenario.output_dir = args.output_directory
            scenario.input_psmac_dirs = args.output_directory
        scenario.shared_model = args.batch
        optimizer = SMAC4MF(scenario=scenario, tae_runner=self.evaluate, n_jobs=-1)
        best_config = optimizer.optimize()
        return best_config, optimizer.get_runhistory(), optimizer.get_trajectory()


def main():
    parser = argparse.ArgumentParser(description='Optimize hopfield-tracking')
    parser.add_argument('--dataset', type=str, default='simple', help='Dataset identifier string',
                        choices=get_datasets())
    parser.add_argument('--batch', action='store_true', help='Share output directory')
    parser.add_argument('--n-events', type=int, default=100,
                        help='Total number of events to read/generate for training.')
    parser.add_argument('--runcount-limit', type=int, default=10,
                        help='Maximum number of runs to perform')
    parser.add_argument('--output-directory', type=str, default=None,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    args = parser.parse_args()

    if args.batch:
        raise ArgumentError(args.shared_directory,
                            'Shared output directory is required when running in a batch system')
    best_config, history, trajectory = Optimizer(args.dataset, args.n_events).run(args)
    print(best_config.get_dictionary())
    event = get_hits(args.dataset, 1)
    seg, acts, positives = iterate.run(event, **best_config)
    tseg = gen_seg_track_layered(event)
    perfect_act = gen_perfect_act(seg, tseg)
    act = acts[-1]
    metrics = metric_history(event, seg, tseg, acts, positives)
    sns.relplot(data=metrics.reset_index().melt('index'), row='variable', x='index', y='value', kind="line",
                facet_kws={'sharey': False, 'sharex': True})
    plt.show()
    df = pd.DataFrame(trajectory).drop(columns='incumbent')
    sns.relplot(data=df.reset_index().melt('index'), row='variable', x='index', y='value', kind="line",
        facet_kws = {'sharey': False, 'sharex': True})
    plt.show()
    df = pd.DataFrame([rec for rec in pd.DataFrame(trajectory).incumbent])
    sns.relplot(data=df.reset_index().melt('index'), row='variable', x='index', y='value', kind="line",
        facet_kws = {'sharey': False, 'sharex': True})
    plt.show()
    plot_result(event, seg, act, perfect_act, positives[-1]).show()
    app.run()


if __name__ == '__main__':
    main()
