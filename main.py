import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import tensor, full

from generator import SimpleEventGenerator
from reconstruct import annealing_curve, dist_act, mean_act, recall, update_layer, should_stop, precision, draw_tracks, \
    draw_tracks_projection, draw_activation_values, plot_activation_hist
from segment import energies as energies_

event = next(SimpleEventGenerator(1).gen_many_events(1, 10))

torch.set_default_tensor_type(torch.cuda.FloatTensor)

pos = [tensor(layer) for layer in event]

act = [full((len(a), len(b)), 0.1, requires_grad=True) for a, b in zip(pos, pos[1:])]

perfect_act = [torch.eye(len(a)) for a in act]

ALPHA = 5.  # наказание форков
BETA = 0.01  # наказание за количество активных
POWER = 5  # степень косинуса в энергии за кривизну
COS_MIN = 0.9  # минимальный косинус за который есть награда

THRESHOLD = 0.5  # граница отсечения активации треков от не-треков

TMAX = 40
TMIN = 0.1
ANNEAL_ITERATIONS = 200
STABLE_ITERATIONS = 200

DROPOUT = 0.5
LEARNING_RATE = 0.2

temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)
plt.plot(temp_curve)

history = pd.DataFrame({'temp': temp_curve})

energies = energies_(pos, ALPHA, BETA, POWER, COS_MIN)

history['energy'] = 0
history['E_curve'] = 0
history['E_fork'] = 0
history['E_number'] = 0
history['dist_perfect'] = 0
history['mean_act'] = 0
history['precision'] = 0
history['recall'] = 0

for i in history.index:
    ec, en, ef = energies(act)
    e_total = ec + ef + en
    history.loc[i, 'energy'] = e_total.item()
    history.loc[i, 'E_curve'] = -ec.item()
    history.loc[i, 'E_fork'] = ef.item()
    history.loc[i, 'E_number'] = en.item()
    history.loc[i, 'dist_perfect'] = dist_act(act, perfect_act)
    history.loc[i, 'mean_act'] = mean_act(act)
    history.loc[i, 'precision'] = precision(act, perfect_act, THRESHOLD)
    history.loc[i, 'recall'] = recall(act, perfect_act, THRESHOLD)

    e_total.backward()
    a_prev = act
    act = [update_layer(a, history.temp[i], DROPOUT, LEARNING_RATE) for a in act]
    if should_stop(a_prev, act):
        history = history.loc[:(i + 1)]
        break

history.plot(y=['precision', 'recall', 'mean_act', 'dist_perfect'], figsize=(12, 12))
history[['E_curve', 'E_fork', 'E_number']].plot(figsize=(12, 12), logy=True)

plot_activation_hist(act)

draw_activation_values(act)

draw_activation_values([a > THRESHOLD for a in act])

draw_tracks(pos, act, perfect_act, THRESHOLD)

draw_tracks_projection(pos, act, perfect_act, THRESHOLD)
