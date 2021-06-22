import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cross import cross_energy_matrix, cross_energy
from curvature import curvature_energy_matrix, curvature_energy
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, draw_tracks, \
    draw_tracks_projection, draw_activation_values, plot_activation_hist, precision, recall, update_layer_grad
from segment import energy_gradient, gen_segments_all
from total import total_activation_matrix, total_activation_energy

n_tracks = 10
df = next(SimpleEventGenerator(1).gen_many_events(1, n_tracks))

pos = df[['x', 'y', 'z']].values

segments = gen_segments_all(df)

act = np.full(sum(len(s) for s in segments), 0.1)

perfect_act = np.concatenate([np.eye(n_tracks).ravel() for _ in segments])

ALPHA = 5.  # наказание форков
BETA = 0.  # наказание за количество активных
BIAS = 0.2 # activation bias, can be used instead of beta
POWER = 5  # степень косинуса в энергии за кривизну
COS_MIN = 0.9  # минимальный косинус за который есть награда

DROP_SELF_ACTIVATION_WEIGHTS = True

THRESHOLD = 0.5  # граница отсечения активации треков от не-треков

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

compute_gradient = energy_gradient(pos, segments, ALPHA, BETA, POWER, COS_MIN, DROP_SELF_ACTIVATION_WEIGHTS)
for i, t in enumerate(temp_curve):
    acts.append(act)
    grad = compute_gradient(act)
    a_prev = act
    act = update_layer_grad(a_prev, grad, t, DROPOUT, LEARNING_RATE, BIAS)
    if np.abs(act - a_prev).sum() < EPS and i > ANNEAL_ITERATIONS:
        break

a, b, c = total_activation_matrix(pos, segments)
crossing_matrix = cross_energy_matrix(segments)
curvature_matrix = curvature_energy_matrix(pos, segments, POWER, COS_MIN)
energy_history = []
for act in acts:
    en = BETA * total_activation_energy(a, b, c, act)
    ef = ALPHA * cross_energy(crossing_matrix, act)
    ec = - curvature_energy(curvature_matrix, act, act)
    energy_history.append([ec, en, ef])
energy_history = pd.DataFrame(energy_history, columns=['E_curve', 'E_number', 'E_fork'])

small_history = pd.DataFrame([
    (
        precision(act, perfect_act, THRESHOLD),
        recall(act, perfect_act, THRESHOLD),
        act.mean(),
        ((act - perfect_act)**2).mean()
    ) for act in acts],
    columns=['precision', 'recall', 'mean_act', 'dist_perfect'])

small_history.plot(figsize=(12, 12))
energy_history.plot(figsize=(12, 12), logy=True)

plot_activation_hist(acts[-1])
draw_activation_values(acts[-1])
draw_activation_values(acts[-1] > THRESHOLD)
draw_tracks(pos, segments, acts[-1], perfect_act, THRESHOLD)
draw_tracks_projection(pos, segments, acts[-1], perfect_act, THRESHOLD)

# for i in range(0, len(acts), 50):
#     plot_activation_hist(acts[i])
#
# for i in range(0, len(acts), 50):
#     draw_activation_values(acts[i])
#
# for i in range(0, len(acts), 50):
#     draw_activation_values([a > THRESHOLD for a in acts[i]])
#
# for i in range(0, len(acts), 50):
#     draw_tracks(pos, acts[i], perfect_act, THRESHOLD)
#
# for i in range(0, len(acts), 50):
#     draw_tracks_projection(pos, acts[i], perfect_act, THRESHOLD)
