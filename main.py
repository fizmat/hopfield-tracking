from statistics import mean

import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts
from matplotlib import pyplot as plt

from cross import cross_energy_matrix, cross_energy_gradient, cross_energy
from curvature import curvature_energy_matrix, curvature_energy_gradient, curvature_energy
from generator import SimpleEventGenerator
from reconstruct import annealing_curve, update_layer_grad, precision, recall, make_tracks_3d
from segment import gen_segments_all
from total import total_activation_matrix, total_activation_energy_gradient, total_activation_energy

N_TRACKS = 2
hits, track_segments = next(SimpleEventGenerator(
    seed=2, field_strength=0.8, noisiness=10, box_size=.5
).gen_many_events(1, N_TRACKS))

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
MIN_ACTIVATION_CHANGE_TO_CONTINUE = 1e-9

temp_curve = annealing_curve(TMIN, TMAX, ANNEAL_ITERATIONS, STABLE_ITERATIONS)
tcdf = pd.DataFrame({'temp': temp_curve})
tcdf.index.name = 'step'
tcdf.plot()

act = np.full(len(seg), THRESHOLD * 0.999)
acts = []

a, b, c = total_activation_matrix(pos, seg, DROP_SELF_ACTIVATION_WEIGHTS)
crossing_matrix = cross_energy_matrix(seg)
curvature_matrix = curvature_energy_matrix(pos, seg, COSINE_POWER, COSINE_MIN, DISTANCE_POWER)
for i, t in enumerate(temp_curve):
    acts.append(act)
    grad = GAMMA * curvature_energy_gradient(curvature_matrix, act) + \
           ALPHA * cross_energy_gradient(crossing_matrix, act) + \
           BETA * total_activation_energy_gradient(a, b, act)
    a_prev = act
    act = update_layer_grad(a_prev, grad, t, DROPOUT, LEARNING_RATE, BIAS)
    if np.abs(act - a_prev).sum() < MIN_ACTIVATION_CHANGE_TO_CONTINUE and i > ANNEAL_ITERATIONS:
        break

energy_history = []
for act in acts:
    en = BETA * total_activation_energy(a, b, c, act)
    ef = ALPHA * cross_energy(crossing_matrix, act)
    ec = - GAMMA * curvature_energy(curvature_matrix, act)
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
small_history.index.name = 'step'

hv.extension('matplotlib')
mr = hv.renderer('matplotlib')

small_history.plot()
plt.show()
energy_history.plot(logy=True)
plt.show()

f, ax = plt.subplots(figsize=(10, 10))
ax.hist(acts[-1])
f.show()

n_steps = 6
steps = np.linspace(0, len(acts) - 1, min(n_steps, len(acts)), dtype=int)

tracks_3d = []
tracks_projection = []
tracks_by_track = []
for i in steps:
    tp, fp, tn, fn = make_tracks_3d(pos, seg, acts[i], perfect_act, THRESHOLD)
    vdims = ['act', 'perfect_act', 'positive', 'true']
    xyz = hv.Overlay([
        hv.Scatter3D(hits[hits.track == -1], kdims=['x', 'y', 'z'], label='noise', group='hits'),
        hv.Scatter3D(hits[hits.track != -1], kdims=['x', 'y', 'z'], label='hits', group='hits'),
        hv.Path3D(tp, vdims=vdims, label='tp', group='tracks'),
        hv.Path3D(fp, vdims=vdims, label='fp', group='tracks'),
        hv.Path3D(fn, vdims=vdims, label='fn', group='tracks')
    ])
    tracks_3d.append(xyz)

    projection = hv.Points(hits[hits.track != -1], kdims=['y', 'z'], label='hits', group='hits') * \
                 hv.Points(hits[hits.track == -1], kdims=['y', 'z'], label='noise', group='hits') * \
                 hv.Path(tp, kdims=['y', 'z'], vdims=vdims, label='tp', group='tracks') * \
                 hv.Path(fp, kdims=['y', 'z'], vdims=vdims, label='fp', group='tracks') * \
                 hv.Path(fn, kdims=['y', 'z'], vdims=vdims, label='fn', group='tracks')
    tracks_projection.append(projection)

    nodes = hv.Nodes(hits, kdims=['track', 'layer', 'index'])
    no_tn = np.logical_or(act > THRESHOLD, perfect_act > THRESHOLD)
    graph = hv.Graph(((*seg[no_tn].transpose(), act[no_tn], perfect_act[no_tn]), nodes), vdims=['act', 'perfect_act'])
    tracks_by_track.append(hv.Overlay([graph]))

track_history = hv.NdLayout(
    {s: tracks_3d[i] for i, s in enumerate(steps)},
    kdims='step'
)
print(track_history)
track_history.opts.info()

mr.show(track_history.opts(
    opts.Scatter3D(alpha=1, color='black'),
    opts.Scatter3D('Hits.Noise', color='black'),
    opts.Path3D(color='black', show_legend=True),
    opts.Path3D('Tracks.fp', color='red'),
    opts.Path3D('Tracks.fn', color='cyan'),
    opts.NdLayout(fig_size=128)
).cols(2))

track_history = hv.NdLayout(
    {s: tracks_projection[i] for i, s in enumerate(steps)},
    kdims='step'
)
print(track_history)
track_history.opts.info()
mr.show(track_history.opts(
    opts.Scatter(alpha=1, color='black'),
    opts.Scatter('Hits.Noise', color='black'),
    opts.Path(color='black', show_legend=True),
    opts.Path('Tracks.fp', color='red'),
    opts.Path('Tracks.fn', color='cyan'),
    opts.NdLayout(fig_size=128)
).cols(2))

track_history = hv.NdLayout(
    {s: tracks_by_track[i] for i, s in enumerate(steps)},
    kdims='step'
)
print(track_history)
track_history.opts.info()
mr.show(track_history.opts(
    opts.Graph(node_size=8, edge_color='act'),
    opts.NdLayout(fig_size=64)
).cols(2))
