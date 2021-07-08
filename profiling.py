#!/usr/bin/env python
# coding: utf-8

from statistics import mean

import numpy as np

from cross import cross_energy_matrix, cross_energy_gradient
from curvature import curvature_energy_matrix, curvature_energy_gradient
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, update_layer_grad
from segment import gen_segments_all
from total import total_activation_matrix, total_activation_energy_gradient

N_TRACKS = 10
N_EVENTS = 10

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

    ALPHA = 0.6  # forks and joins

    BETA = 0  # total activation
    DROP_SELF_ACTIVATION_WEIGHTS = True  # in total activation matrix
    BIAS = 0.2  # activation bias, instead of total activation matrix

    COSINE_POWER = 5
    COSINE_MIN = 0.7
    DISTANCE_POWER = 1
    GAMMA = mean((xa - xb) * (xb - xc)
                 for xa, la in hits[['x', 'layer']].values
                 for xb in hits[hits.layer == la + 1].x
                 for xc in hits[hits.layer == la + 2].x) ** DISTANCE_POWER

    THRESHOLD = 0.5  # activation threshold for segment classification

    TMAX = 20
    TMIN = 0.5
    ANNEAL_ITERATIONS = 40
    STABLE_ITERATIONS = 200

    DROPOUT = 0.5
    LEARNING_RATE = 0.6
    MIN_ACTIVATION_CHANGE_TO_CONTINUE = 0

    a, b, c = total_activation_matrix(pos, seg, DROP_SELF_ACTIVATION_WEIGHTS)
    crossing_matrix = cross_energy_matrix(seg)
    curvature_matrix = curvature_energy_matrix(pos, seg, COSINE_POWER, COSINE_MIN, DISTANCE_POWER)

    temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)

    act = np.full(len(seg), THRESHOLD * 0.999)
    acts = []
    for i, t in enumerate(temp_curve):
        acts.append(act)
        grad = GAMMA * curvature_energy_gradient(curvature_matrix, act) + \
               ALPHA * cross_energy_gradient(crossing_matrix, act) + \
               BETA * total_activation_energy_gradient(a, b, act)
        a_prev = act
        act = update_layer_grad(a_prev, grad, t, DROPOUT, LEARNING_RATE, BIAS)
        if np.abs(act - a_prev).sum() < MIN_ACTIVATION_CHANGE_TO_CONTINUE and i > ANNEAL_ITERATIONS:
            break
