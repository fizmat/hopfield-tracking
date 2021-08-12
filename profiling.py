#!/usr/bin/env python
# coding: utf-8

import numpy as np
# from memory_profiler import profile

from cross import segment_forks, segment_joins, segment_kinks
from curvature import curvature_energy_matrix, segment_adjacent_pairs
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, update_layer_grad, energy_gradient
from segment import gen_segments_all

N_TRACKS = 10
N_EVENTS = 500


# @profile
def main():
    eventgen = SimpleEventGenerator(
        seed=1, field_strength=0.8, noisiness=0, box_size=.5
    ).gen_many_events(N_EVENTS, N_TRACKS)

    for hits, track_segments in eventgen:
        pos = hits[['x', 'y', 'z']].values

        seg = gen_segments_all(hits)

        perfect_act = np.zeros(len(seg))
        track_segment_set = set(tuple(s) for s in track_segments)
        is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
        perfect_act[is_in_track] = 1

        ALPHA = 0.5  # forks and joins

        BETA = 0  # total activation
        DROP_SELF_ACTIVATION_WEIGHTS = True  # in total activation matrix
        BIAS = 0.4  # activation bias, instead of total activation matrix

        COSINE_POWER = 5
        COSINE_MIN = 0.8
        DISTANCE_POWER = 0.5
        LZ = hits.groupby('layer').z.mean()
        l_dist = (LZ.values[1:] - LZ.values[:-1]).mean()
        GAMMA = l_dist ** DISTANCE_POWER

        THRESHOLD = 0.5  # activation threshold for segment classification

        TMAX = 5
        TMIN = 0.2
        ANNEAL_ITERATIONS = 200
        STABLE_ITERATIONS = 200

        DROPOUT = 0
        LEARNING_RATE = 0.1
        MIN_ACTIVATION_CHANGE_TO_CONTINUE = 0
        SHOULD_STOP_LOOKBACK = 7

        pairs = segment_adjacent_pairs(seg)
        forks = segment_forks(seg)
        joins = segment_joins(seg)
        kinks = segment_kinks(seg, pos, COSINE_MIN, pairs)
        crossing_matrix = forks + joins + kinks
        curvature_matrix = curvature_energy_matrix(pos, seg, pairs, COSINE_POWER, COSINE_MIN,
                                                   DISTANCE_POWER)
        e_matrix = ALPHA / 2 * crossing_matrix - GAMMA / 2 * curvature_matrix

        temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)

        act = np.full(len(seg), 0.9)
        for i, t in enumerate(temp_curve):
            grad = energy_gradient(e_matrix, act)
            update_layer_grad(act, grad, t, DROPOUT, LEARNING_RATE, BIAS)
            # if i > ANNEAL_ITERATIONS and should_stop(act, acts, MIN_ACTIVATION_CHANGE_TO_CONTINUE, SHOULD_STOP_LOOKBACK):
            #     break


if __name__ == '__main__':
    main()
