#!/usr/bin/env python
# coding: utf-8

import numpy as np

from cross import cross_energy_matrix
from curvature import curvature_energy_matrix, segment_adjacent_pairs
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, update_layer_grad, energy_gradient, should_stop
from segment import gen_segments_all
from total import total_activation_matrix

N_TRACKS = 10
N_EVENTS = 500

eventgen = SimpleEventGenerator(
    seed=2, field_strength=0.8, noisiness=10, box_size=.5
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

    crossing_matrix = cross_energy_matrix(seg) if ALPHA else 0
    a, b, c = total_activation_matrix(pos, seg, DROP_SELF_ACTIVATION_WEIGHTS) if BETA else (0, 0, 0)
    pairs = segment_adjacent_pairs(seg)
    curvature_matrix = curvature_energy_matrix(pos, seg, pairs, COSINE_POWER, COSINE_MIN, DISTANCE_POWER) if GAMMA else 0
    e_matrix = ALPHA / 2 * crossing_matrix + BETA / 2 * a - GAMMA / 2 * curvature_matrix
    if BETA:
        e_matrix = e_matrix.A  # dense matrix to ndarray for correct .dot behaviour (same as sparse matrix)

    temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)

    act = np.full(len(seg), THRESHOLD * 0.999)
    acts = []
    for i, t in enumerate(temp_curve):
        acts.append(act.copy())
        grad = energy_gradient(e_matrix, act) + (b if BETA else 0)
        update_layer_grad(act, grad, t, DROPOUT, LEARNING_RATE, BIAS)
        if i > ANNEAL_ITERATIONS and should_stop(act, acts, MIN_ACTIVATION_CHANGE_TO_CONTINUE, SHOULD_STOP_LOOKBACK):
            break
