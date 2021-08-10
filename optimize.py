#!/usr/bin/env python
# coding: utf-8

import ConfigSpace as CS
import numpy as np
from hpbandster.core.nameserver import NameServer
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from sklearn.metrics import roc_auc_score

from cross import cross_energy_matrix
from curvature import curvature_energy_matrix, segment_adjacent_pairs
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, update_layer_grad, energy_gradient, should_stop
from segment import gen_segments_all

N_TRACKS = 10
N_EVENTS = 500


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch = list(SimpleEventGenerator(
            seed=2, field_strength=0.8, noisiness=10, box_size=.5
        ).gen_many_events(N_EVENTS, N_TRACKS))

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
        batch = self.batch[:int(budget)]
        total = 0.
        for hits, track_segments in batch:
            pos = hits[['x', 'y', 'z']].values

            seg = gen_segments_all(hits)

            perfect_act = np.zeros(len(seg))
            track_segment_set = set(tuple(s) for s in track_segments)
            is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
            perfect_act[is_in_track] = 1

            crossing_matrix = cross_energy_matrix(seg)
            pairs = segment_adjacent_pairs(seg)
            curvature_matrix = curvature_energy_matrix(pos, seg, pairs,
                                                       config['cosine_power'], config['cosine_min'],
                                                       config['distance_power'])
            e_matrix = config['alpha'] / 2 * crossing_matrix - config['gamma'] / 2 * curvature_matrix
            tmin = 1.
            temp_curve = annealing_curve(tmin, config['tmax'], config['anneal_steps'], config['stable_steps'])

            act = np.full(len(seg), config['starting_act'])
            for i, t in enumerate(temp_curve):
                grad = energy_gradient(e_matrix, act)
                update_layer_grad(act, grad, t, config['dropout'], config['learning_rate'], config['bias'])
            total += roc_auc_score(perfect_act, act)
        loss = 1 - total / budget


        return ({
            'loss': loss,
            'info': {},
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('alpha', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('gamma', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bias', lower=-10, upper=10))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_power', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('cosine_min', lower=0, upper=20))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('distance_power', lower=0, upper=3))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('tmax', lower=1, upper=100))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('anneal_steps', lower=2, upper=1000))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('stable_steps', lower=2, upper=1000))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('starting_act', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('dropout', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0, upper=1))
        return (config_space)


def main():
    ns = NameServer(run_id='example1', host='127.0.0.1', port=2222)
    ns.start()
    w = MyWorker(nameserver='127.0.0.1', nameserver_port=2222, run_id='example1')
    w.run(background=True)
    bohb = BOHB(configspace=w.get_configspace(),
                nameserver='127.0.0.1', nameserver_port=2222, run_id='example1',
                min_budget=10, max_budget=N_EVENTS
                )
    res = bohb.run(n_iterations=2)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (
            sum([r.budget for r in res.get_all_runs()]) / N_EVENTS))


if __name__ == '__main__':
    main()
