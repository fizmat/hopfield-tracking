import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from generator import SimpleEventGenerator
from reconstruct import annealing_curve, should_stop, draw_tracks, \
    draw_tracks_projection, draw_activation_values, plot_activation_hist, precision, recall, mean_act, dist_act, \
    update_layer_grad
from segment import energies as energies_, energy_gradient, gen_segments_all

df = next(SimpleEventGenerator(1).gen_many_events(1, 10))

pos = df[['x', 'y', 'z']].values

segments = gen_segments_all(df)

act = [np.full(s.shape[-1], 0.1) for s in segments]

perfect_act = [np.eye(10).ravel() for a in act]

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

temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)
plt.plot(temp_curve)

acts = []

compute_gradient = energy_gradient(pos, segments, ALPHA, BETA, POWER, COS_MIN, DROP_SELF_ACTIVATION_WEIGHTS)
for i, t in enumerate(temp_curve):
    acts.append(act)
    grad = compute_gradient(act)
    a_prev = act
    act = [update_layer_grad(a, g, t, DROPOUT, LEARNING_RATE, BIAS) for a, g in zip(a_prev, grad)]
    if should_stop(a_prev, act) and i > ANNEAL_ITERATIONS:
        break

energies = energies_(pos, segments, ALPHA, BETA, POWER, COS_MIN)
energy_history = []
for act in acts:
    ec, en, ef = energies(act)
    ec = -ec
    energy_history.append([ec.item(), en.item(), ef.item()])
energy_history = pd.DataFrame(energy_history, columns=['E_curve', 'E_number', 'E_fork'])

small_history = pd.DataFrame([
    (
        precision(act, perfect_act, THRESHOLD),
        recall(act, perfect_act, THRESHOLD),
        mean_act(act),
        dist_act(act, perfect_act)
    ) for act in acts],
    columns=['precision', 'recall', 'mean_act', 'dist_perfect'])

small_history.plot(figsize=(12, 12))
energy_history.plot(figsize=(12, 12), logy=True)

plot_activation_hist(acts[-1])
draw_activation_values(acts[-1])
draw_activation_values([a > THRESHOLD for a in acts[-1]])
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
