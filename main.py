import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cross import cross_energy_matrix, cross_energy_gradient, cross_energy
from curvature import curvature_energy_matrix, curvature_energy_gradient, curvature_energy
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, update_layer_grad, draw_tracks, \
    draw_tracks_projection, precision, recall
from segment import gen_segments_all
from total import total_activation_matrix, total_activation_energy_gradient, total_activation_energy

n_tracks = 10
hits, track_segments = next(SimpleEventGenerator(
    seed=2, field_strength=0.8, noisiness=10, box_size=.5
).gen_many_events(1, n_tracks))

pos = hits[['x', 'y', 'z']].values

seg = gen_segments_all(hits)

act = np.full(len(seg), 0.5)

perfect_act = np.zeros_like(act)

track_segment_set = set(tuple(s) for s in track_segments)
is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
perfect_act[is_in_track] = 1

ALPHA = 5.  # наказание форков
BETA = 0.  # наказание за количество активных
BIAS = 0.2  # activation bias, can be used instead of beta
POWER = 5  # степень косинуса в энергии за кривизну
COS_MIN = 0.9  # минимальный косинус за который есть награда

DROP_SELF_ACTIVATION_WEIGHTS = True

THRESHOLD = 0.3  # граница отсечения активации треков от не-треков

TMAX = 40
TMIN = 0.1
ANNEAL_ITERATIONS = 200
STABLE_ITERATIONS = 200

DROPOUT = 0.5
LEARNING_RATE = 0.2
EPS = 1e-4

temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)
plt.plot(temp_curve)

acts = []

a, b, c = total_activation_matrix(pos, seg, DROP_SELF_ACTIVATION_WEIGHTS)
crossing_matrix = cross_energy_matrix(seg)
curvature_matrix = curvature_energy_matrix(pos, seg, POWER, COS_MIN)
for i, t in enumerate(temp_curve):
    acts.append(act)
    grad = curvature_energy_gradient(curvature_matrix, act) + \
           ALPHA * cross_energy_gradient(crossing_matrix, act) + \
           BETA * total_activation_energy_gradient(a, b, act)
    a_prev = act
    act = update_layer_grad(a_prev, grad, t, DROPOUT, LEARNING_RATE, BIAS)
    if np.abs(act - a_prev).sum() < EPS and i > ANNEAL_ITERATIONS:
        break

energy_history = []
for act in acts:
    en = BETA * total_activation_energy(a, b, c, act)
    ef = ALPHA * cross_energy(crossing_matrix, act)
    ec = - curvature_energy(curvature_matrix, act)
    energy_history.append([ec, en, ef])
energy_history = pd.DataFrame(energy_history, columns=['E_curve', 'E_number', 'E_fork'])

small_history = pd.DataFrame([
    (
        precision(act, perfect_act, THRESHOLD),
        recall(act, perfect_act, THRESHOLD),
        act.mean(),
        ((act - perfect_act) ** 2).mean()
    ) for act in acts],
    columns=['precision', 'recall', 'mean_act', 'dist_perfect'])

small_history.plot(figsize=(12, 12))
plt.show()
energy_history.plot(figsize=(12, 12), logy=True)
plt.show()

f, ax = plt.subplots(figsize=(10,10))
ax.hist(acts[-1])
f.show()
draw_tracks(pos, seg, acts[-1], perfect_act, THRESHOLD)
draw_tracks_projection(pos, seg, acts[-1], perfect_act, THRESHOLD)

for i in range(0, len(acts), 50):
    f, ax = plt.subplots(figsize=(10, 10))
    ax.hist(acts[i])
    f.show()
for i in range(0, len(acts), 50):
    draw_tracks(pos, seg, acts[i], perfect_act, THRESHOLD)
for i in range(0, len(acts), 50):
    draw_tracks_projection(pos, seg, acts[i], perfect_act, THRESHOLD)
