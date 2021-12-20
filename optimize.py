#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import pickle
import socket
import time

import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
import numpy as np
import pandas as pd
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from sklearn.metrics import f1_score

from cross import cross_energy_matrix
from curvature import curvature_energy_matrix, segment_adjacent_pairs
from datasets import get_hits_trackml_by_module, get_hits_bman, get_hits_simple
from reconstruct import annealing_curve, update_layer_grad, energy_gradient
from metrics.tracks import build_segmented_tracks, found_tracks, found_crosses
from segment import gen_segments_all

logging.basicConfig(level=logging.WARNING)


def mark_track_segments(hits):
    track_segments = []
    for track, g in hits.groupby('track'):
        if track >= 0:
            for i in range(min(g.layer), max(g.layer)):
                for a in g[g.layer == i].index:
                    for b in g[g.layer == i + 1].index:
                        track_segments.append((a, b))
    return track_segments


class MyWorker(Worker):
    def __init__(self, max_hits, n_events, total_steps, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_hits = max_hits
        self.total_steps = total_steps
        if dataset.lower() == 'bman':
            hits = get_hits_bman(max_hits)
        elif dataset.lower() == 'simple':
            hits = get_hits_simple()
        elif dataset.lower() == 'trackml':
            hits = get_hits_trackml_by_module()
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        events = hits.event_id.unique()
        sample = np.random.choice(events, size=n_events * 2, replace=False)

        train, test = sample[:n_events], sample[n_events:]
        train_hits = [hits[hits.event_id == event].reset_index(drop=True) for event in train]
        test_hits = [hits[hits.event_id == event].reset_index(drop=True) for event in test]
        self.train_batch = [(h, mark_track_segments(h)) for h in train_hits]
        self.test_batch = [(h, mark_track_segments(h)) for h in test_hits]

    def compute(self, config, budget, **kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        score = []
        for batch in (self.train_batch[:int(budget)], self.test_batch[:int(budget)]):
            reds = 0
            tracks = 0
            crosses = 0
            for hits, track_segments in batch:
                pos = hits[['x', 'y', 'z']].values

                seg = gen_segments_all(hits)

                perfect_act = np.zeros(len(seg))
                track_segment_set = set(tuple(s) for s in track_segments)
                is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
                if len(is_in_track):
                    perfect_act[is_in_track] = 1

                pairs = segment_adjacent_pairs(seg)
                crossing_matrix = cross_energy_matrix(seg, pos, config['cosine_min_allowed'], pairs)
                curvature_matrix = curvature_energy_matrix(pos, seg, pairs,
                                                           config['cosine_power'], config['cosine_min_rewarded'],
                                                           config['distance_power'])
                e_matrix = config['alpha'] / 2 * crossing_matrix - config['gamma'] / 2 * curvature_matrix
                tmin = config['tmin']
                temp_curve = annealing_curve(tmin, config['tmax'],
                                             config['anneal_steps'], config['total_steps'] - config['anneal_steps'])

                act = np.full(len(seg), config['starting_act'])
                for i, t in enumerate(temp_curve):
                    grad = energy_gradient(e_matrix, act)
                    update_layer_grad(act, grad, t, config['dropout'], config['learning_rate'], config['bias'])
                reds += np.sum((act > config['threshold']) & (perfect_act < config['threshold']))
                segmented_tracks = build_segmented_tracks(hits).values()
                tracks += found_tracks(seg, act, segmented_tracks)
                crosses += found_crosses(seg, act)
            reds, tracks, crosses = (int(x) for x in (reds, tracks, crosses))
            score.append({'reds': reds, 'tracks': tracks, 'crosses': crosses, 'loss': -(tracks-crosses-0.036*reds)})

        return ({
            'loss': score[0]['loss'],
            'info': {
                "train_score": score[0],
                "test_score": score[1]
            }
        })

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
        config_space.add_hyperparameter(CS.Constant('max_hits', value=self.max_hits))
        config_space.add_hyperparameter(CS.Constant('total_steps', value=self.total_steps))
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter('anneal_steps', lower=0, upper=self.total_steps))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('starting_act', lower=0, upper=1))
        config_space.add_hyperparameter(CS.Constant('dropout', value=0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0, upper=1))
        return config_space


def test(dataset='BMaN'):
    worker = MyWorker(run_id='0', max_hits=100, n_events=2, total_steps=10, dataset=dataset)
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2, working_directory='workdir')
    print(res)


def main():
    parser = argparse.ArgumentParser(description='Optimize hopfield-tracking')
    parser.add_argument('--test', help='Flag to run worker once locally', action='store_true')
    parser.add_argument('--max_hits', type=int, help='Max number of hits per event (memory limits)', default=500)
    parser.add_argument('--min_budget', type=int, help='Minimum budget (in events) used during the optimization.',
                        default=1)
    parser.add_argument('--max_budget', type=int, help='Maximum budget (in events) used during the optimization.',
                        default=10)
    parser.add_argument('--hopfield_steps', type=int, help='Total length of iteration in anneal and post-anneal',
                        default=10)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. \
                              An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=1)
    parser.add_argument('--shared_directory', type=str, default='workdir',
                        help='A directory that is accessible for all processes, e.g. a NFS share.')
    parser.add_argument('--dataset', type=str, help='Dataset identifier string', default='simple')

    args = parser.parse_args()

    if args.test:
        test(args.dataset)
        exit(0)

    host = socket.gethostname()

    if args.worker:
        time.sleep(60)
        w = MyWorker(max_hits=args.max_hits, n_events=args.max_budget, total_steps=args.hopfield_steps,
                     run_id=args.run_id, host=host, dataset=args.dataset)
        w.load_nameserver_credentials(working_directory=args.shared_directory)
        w.run(background=False)
        exit(0)

    ns = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
    ns_host, ns_port = ns.start()
    w = MyWorker(max_hits=args.max_hits, n_events=args.max_budget, run_id=args.run_id, total_steps=args.hopfield_steps,
                 host=host, nameserver=ns_host, nameserver_port=ns_port, dataset=args.dataset)
    w.run(background=True)

    bohb = BOHB(configspace=w.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=args.min_budget, max_budget=args.max_budget
                )
    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

    with open(os.path.join(args.shared_directory, f'{args.run_id}.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()


if __name__ == '__main__':
    main()
