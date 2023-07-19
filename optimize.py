import argparse
import logging
from argparse import ArgumentError
from typing import Dict, Tuple

import ConfigSpace as CS
import numpy as np
import pandas as pd
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

from datasets import get_hits, get_datasets
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import segment_adjacent_pairs, curvature_energy_matrix
from hopfield.iterate import annealing_curve, hopfield_history
from metrics.tracks import track_metrics
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_layered

logging.basicConfig(level=logging.WARNING)


class MyWorker:
    def __init__(self, n_events, total_steps, dataset, *args, **kwargs):
        self.total_steps = total_steps
        self.dataset = dataset.lower()
        hits = get_hits(self.dataset, n_events=n_events * 2)
        events = hits.event_id.unique()
        self.train_batch = [hits[hits.event_id == event] for event in events[:n_events]]
        self.test_batch = [hits[hits.event_id == event] for event in events[n_events:]]

    def compute(self, config: Dict, instance, seed: int) -> Tuple[float, Dict]:
        score = []
        for batch in (self.train_batch, self.test_batch):
            event_metrics = []
            for hits in batch:
                hits.reset_index(drop=True, inplace=True)
                pos = hits[['x', 'y', 'z']].to_numpy()
                seg = gen_seg_layered(hits)
                pairs = segment_adjacent_pairs(seg)
                crossing_matrix = cross_energy_matrix(seg)
                curvature_matrix = curvature_energy_matrix(pos, seg, pairs, config['alpha'], config['gamma'],
                                                           config['cosine_power'], config['cosine_min_rewarded'],
                                                           config['distance_power'], config['cosine_min_allowed'])
                energy_matrix = crossing_matrix + curvature_matrix
                temp_curve = annealing_curve(config['tmin'], config['tmax'], config['anneal_steps'],
                                             config['total_steps'] - config['anneal_steps'])
                starting_act = np.full(len(seg), config['starting_act'])
                act = hopfield_history(energy_matrix, temp_curve, starting_act, config['dropout'], config['learning_rate'],
                                       config['bias'])[-1]
                tseg = gen_seg_track_layered(hits)
                event_metrics.append(track_metrics(hits, seg, tseg, act, config['threshold']))
            event_metrics = pd.DataFrame(event_metrics)
            score.append(event_metrics.mean().to_dict())
        return (
            1. - score[0]['trackml'],
            {
                "train_score": score[0],
                "test_score": score[1]
            }
        )

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
        config_space.add_hyperparameter(CS.Constant('tmin', value=1.))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('tmax', lower=1, upper=100))
        config_space.add_hyperparameter(CS.Constant('total_steps', value=self.total_steps))
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter('anneal_steps', lower=0, upper=self.total_steps))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('starting_act', lower=0, upper=1))
        config_space.add_hyperparameter(CS.Constant('dropout', value=0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0, upper=1))
        return config_space


def run(args):
    worker = MyWorker(n_events=args.max_budget, total_steps=args.hopfield_iterations, dataset=args.dataset)
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": args.runcount_limit,
        'cs': worker.get_configspace(),
    })

    if args.output_directory:
        scenario.output_dir = args.output_directory
        scenario.input_psmac_dirs = args.output_directory
    scenario.shared_model = args.batch


    optimizer = SMAC4BB(scenario=scenario, tae_runner=worker.compute)
    best_config = optimizer.optimize()
    return best_config, optimizer.get_runhistory(), optimizer.get_trajectory()


def main():
    parser = argparse.ArgumentParser(description='Optimize hopfield-tracking')
    parser.add_argument('--dataset', type=str, default='simple', help='Dataset identifier string',
                        choices=get_datasets())
    parser.add_argument('--batch', action='store_true', help='Share output directory')
    parser.add_argument('--min-budget', type=int, default=1,
                        help='Minimum budget (in events) used during the optimization.')
    parser.add_argument('--max-budget', type=int, default=10,
                        help='Maximum budget (in events) used during the optimization.')
    parser.add_argument('--hopfield-iterations', type=int, default=10,
                        help='Total number of iteration in the anneal and post-anneal phases')
    parser.add_argument('--runcount-limit', type=int, default=10,
                        help='Maximum number of runs to perform')
    parser.add_argument('--output-directory', type=str, default=None,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    args = parser.parse_args()

    if args.batch:
        raise ArgumentError(args.shared_directory,
                            'Shared output directory is required when running in a batch system')
    best_config, history, trajectory = run(args)
    print(pd.DataFrame([best_config]).T)
    print(pd.DataFrame(trajectory).drop(columns='incumbent').T)
    for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in history.data.items():
        print(config_id, instance_id, seed, budget)
        print(cost, time, status, starttime, endtime, additional_info)


if __name__ == '__main__':
    main()
