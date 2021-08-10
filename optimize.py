#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import pickle
import time

import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
import numpy as np
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from sklearn.metrics import roc_auc_score

from cross import cross_energy_matrix
from curvature import curvature_energy_matrix, segment_adjacent_pairs
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, update_layer_grad, energy_gradient
from segment import gen_segments_all

logging.basicConfig(level=logging.WARNING)


class MyWorker(Worker):

    def __init__(self, n_events, n_tracks=10, field_strength=0.8, noisiness=10, noise_box=.5, train_seed=1, validation_seed=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_batch = list(SimpleEventGenerator(
            seed=train_seed, field_strength=field_strength, noisiness=noisiness, box_size=noise_box
        ).gen_many_events(n_events, n_tracks))
        self.validation_batch = list(SimpleEventGenerator(
            seed=validation_seed, field_strength=field_strength, noisiness=noisiness, box_size=noise_box
        ).gen_many_events(n_events, n_tracks))

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
        loss = []
        for batch in (self.train_batch[:int(budget)], self.validation_batch[:int(budget)]):
            total = 0.
            for hits, track_segments in batch:
                pos = hits[['x', 'y', 'z']].values

                seg = gen_segments_all(hits)

                perfect_act = np.zeros(len(seg))
                track_segment_set = set(tuple(s) for s in track_segments)
                is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
                perfect_act[is_in_track] = 1

                crossing_matrix = cross_energy_matrix(seg, pos, config['cosine_min_allowed'])
                pairs = segment_adjacent_pairs(seg)
                curvature_matrix = curvature_energy_matrix(pos, seg, pairs,
                                                           config['cosine_power'], config['cosine_min_rewarded'],
                                                           config['distance_power'])
                e_matrix = config['alpha'] / 2 * crossing_matrix - config['gamma'] / 2 * curvature_matrix
                tmin = 1.
                temp_curve = annealing_curve(tmin, config['tmax'], config['anneal_steps'], config['stable_steps'])

                act = np.full(len(seg), config['starting_act'])
                for i, t in enumerate(temp_curve):
                    grad = energy_gradient(e_matrix, act)
                    update_layer_grad(act, grad, t, config['dropout'], config['learning_rate'], config['bias'])
                total += roc_auc_score(perfect_act, act)
            loss.append(1 - total / budget)

        return ({
            'loss': loss[0],
            'info': {"validation_loss": loss[1]},
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('alpha', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('gamma', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bias', lower=-10, upper=10))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_power', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_min_allowed', lower=-1, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_min_rewarded', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('distance_power', lower=0, upper=3))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('tmax', lower=1, upper=100))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('anneal_steps', lower=2, upper=1000))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('stable_steps', lower=2, upper=1000))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('starting_act', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('dropout', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0, upper=1))
        return (config_space)


def test():
    worker = MyWorker(run_id='0', n_events=2)
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2, working_directory='.')
    print(res)


def main():
    parser = argparse.ArgumentParser(description='Optimize hopfield-tracking')
    parser.add_argument('--test', help='Flag to run worker once locally', action='store_true')
    parser.add_argument('--n_tracks', type=int, help='number of tracks per event', default=10)
    parser.add_argument('--min_budget', type=int, help='Minimum budget (in events) used during the optimization.', default=10)
    parser.add_argument('--max_budget', type=int, help='Maximum budget (in events) used during the optimization.', default=500)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. \
                              An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.')
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    args = parser.parse_args()

    if args.test:
         test()
         exit(0)

    host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:
        time.sleep(60)
        w = MyWorker(n_tracks=args.n_tracks, n_events=args.max_budget, run_id=args.run_id, host=host)
        w.load_nameserver_credentials(working_directory=args.shared_directory)
        w.run(background=False)
        exit(0)

    ns = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
    ns_host, ns_port = ns.start()
    w = MyWorker(n_tracks=args.n_tracks, n_events=args.max_budget, run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    bohb = BOHB(configspace=MyWorker.get_configspace(),
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
