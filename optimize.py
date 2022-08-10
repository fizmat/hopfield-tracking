#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import pickle
import socket
import time
from argparse import ArgumentError
from datetime import datetime

import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
import numpy as np
import pandas as pd
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from pathos.multiprocessing import ProcessPool

from datasets import get_hits
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import segment_adjacent_pairs, curvature_energy_matrix
from hopfield.iterate import annealing_curve, hopfield_history
from metrics.tracks import track_metrics
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_layered

logging.basicConfig(level=logging.WARNING)


class MyWorker(Worker):
    def __init__(self, n_events, total_steps, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = total_steps
        self.dataset = dataset.lower()
        hits = get_hits(self.dataset, n_events=n_events * 2)
        events = hits.event_id.unique()
        self.train_batch = [hits[hits.event_id == event] for event in events[:n_events]]
        self.test_batch = [hits[hits.event_id == event] for event in events[n_events:]]

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
            event_metrics['loss'] = 0
            score.append(event_metrics.sum().to_dict())

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
        config_space.add_hyperparameter(CS.Constant('total_steps', value=self.total_steps))
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter('anneal_steps', lower=0, upper=self.total_steps))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('starting_act', lower=0, upper=1))
        config_space.add_hyperparameter(CS.Constant('dropout', value=0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0, upper=1))
        return config_space


def run(args):
    worker = MyWorker(run_id=0, n_events=args.max_budget, total_steps=args.hopfield_steps, dataset=args.dataset)
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2, working_directory='workdir')
    print(res)


def worker(args):
    host = socket.gethostname()
    time.sleep(args.worker_delay)  # wait to make sure master is online
    w = MyWorker(n_events=args.max_budget, total_steps=args.hopfield_steps,
                 run_id=args.run_id, host=host, dataset=args.dataset)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)


def master(args):
    host = socket.gethostname()
    ns = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
    ns_host, ns_port = ns.start()
    w = MyWorker(n_events=args.max_budget, run_id=args.run_id, total_steps=args.hopfield_steps,
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


def main():
    parser = argparse.ArgumentParser(description='Optimize hopfield-tracking')
    parser.add_argument('mode', help='Run mode', type=str,
                        choices=['sequential', 'parallel', 'worker', 'master'])
    parser.add_argument('dataset', type=str, help='Dataset identifier string')
    parser.add_argument('--min_budget', type=int, help='Minimum budget (in events) used during the optimization.',
                        default=1)
    parser.add_argument('--max_budget', type=int, help='Maximum budget (in events) used during the optimization.',
                        default=10)
    parser.add_argument('--hopfield_steps', type=int, help='Total length of iteration in anneal and post-anneal',
                        default=10)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
    parser.add_argument('--worker_delay', type=float, help='Worker delay in seconds before connecting to master')
    parser.add_argument('--run_id', type=str, help='Unique run id for this optimization run.')
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=1)
    parser.add_argument('--shared_directory', type=str, default='workdir',
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    args = parser.parse_args()

    if args.mode == 'sequential':
        if args.run_id is None:
            args.run_id = datetime.now().strftime('%Y-%m-%d_%H-%M')
        run(args)
    elif args.mode == 'worker':
        if args.run_id is None:
            raise ArgumentError(args.run_id, 'run_id is required in worker mode')
        if args.worker_delay is None:
            args.worker_delay = 60
        worker(args)
    elif args.mode == 'master':
        if args.run_id is None:
            raise ArgumentError(args.run_id, 'run_id is required in master mode')
        master(args)
    elif args.mode == 'parallel':
        if args.worker_delay is None:
            args.worker_delay = 5
        if args.run_id is None:
            args.run_id = datetime.now().strftime('%Y-%m-%d_%H-%M')
        with ProcessPool(nodes=args.n_workers) as pool:
            pool.amap(worker, [args] * args.n_workers)
            master(args)


if __name__ == '__main__':
    main()
