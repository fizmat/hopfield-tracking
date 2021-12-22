import numpy as np
# from memory_profiler import profile

from cross import cross_energy_matrix
from curvature import segment_adjacent_pairs, curvature_energy_matrix
from reconstruct import annealing_curve, energy_gradient, update_layer_grad


# @profile
def hopfield_iterate(config, pos, seg):
    pairs = segment_adjacent_pairs(seg)
    crossing_matrix = cross_energy_matrix(seg, pos, config['cosine_min_allowed'], pairs)
    curvature_matrix = curvature_energy_matrix(pos, seg, pairs,
                                               config['cosine_power'], config['cosine_min_rewarded'],
                                               config['distance_power'])
    e_matrix = config['alpha'] / 2 * crossing_matrix - config['gamma'] / 2 * curvature_matrix
    tmin = config['tmin']
    temp_curve = annealing_curve(tmin, config['tmax'],
                                 config['anneal_steps'], config['total_steps'] - config['anneal_steps'])
    act = np.full(len(seg), config['starting_act'])
    for i, t in enumerate(temp_curve):
        grad = energy_gradient(e_matrix, act)
        update_layer_grad(act, grad, t, config['dropout'], config['learning_rate'], config['bias'])
    return act