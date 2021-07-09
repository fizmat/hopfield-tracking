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

    ALPHA = 0.6  # forks and joins

    BETA = 0  # total activation
    DROP_SELF_ACTIVATION_WEIGHTS = True  # in total activation matrix
    BIAS = 0.2  # activation bias, instead of total activation matrix

    COSINE_POWER = 5
    COSINE_MIN = 0.7
    DISTANCE_POWER = 1
    LX = hits.groupby('layer').x.mean()
    l_dist = (LX.values[1:] - LX.values[:-1]).mean()
    GAMMA = l_dist ** DISTANCE_POWER

    THRESHOLD = 0.5  # activation threshold for segment classification

    TMAX = 20
    TMIN = 0.5
    ANNEAL_ITERATIONS = 40
    STABLE_ITERATIONS = 200

    DROPOUT = 0.5
    LEARNING_RATE = 0.6
    MIN_ACTIVATION_CHANGE_TO_CONTINUE = 0

    crossing_matrix = cross_energy_matrix(seg) if ALPHA else 0
    a, b, c = total_activation_matrix(pos, seg, DROP_SELF_ACTIVATION_WEIGHTS) if BETA else (0, 0, 0)
    curvature_matrix = curvature_energy_matrix(pos, seg, COSINE_POWER, COSINE_MIN, DISTANCE_POWER) if GAMMA else 0

    temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)

    act = np.full(len(seg), THRESHOLD * 0.999)
    acts = []
    for i, t in enumerate(temp_curve):
        acts.append(act)
        grad = (ALPHA * cross_energy_gradient(crossing_matrix, act) if ALPHA else 0) + \
               (BETA * total_activation_energy_gradient(a, b, act) if BETA else 0) + \
               (GAMMA * curvature_energy_gradient(curvature_matrix, act) if GAMMA else 0)
        a_prev = act
        act = update_layer_grad(a_prev, grad, t, DROPOUT, LEARNING_RATE, BIAS)
        if np.abs(act - a_prev).sum() < MIN_ACTIVATION_CHANGE_TO_CONTINUE and i > ANNEAL_ITERATIONS:
            break
