from typing import List, Union, Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix


def annealing_curve(t_min, t_max, cooling_steps, rest_steps):
    return np.concatenate([
        np.geomspace(t_max, t_min, cooling_steps),
        np.full(rest_steps, t_min)])


def update_layer_grad(act: ndarray, grad: ndarray, t: float, dropout_rate: float = 0.,
                      learning_rate: float = 1., bias: float = 0.) -> None:
    n = len(act)
    if dropout_rate:
        not_dropout = np.random.choice(n, round(n * (1. - dropout_rate)), replace=False)
        next_act = 0.5 * (1 + np.tanh((- grad[not_dropout] + bias) / t))
        updated_act = next_act * learning_rate + act[not_dropout] * (1. - learning_rate)
        act[not_dropout] = updated_act
    else:
        next_act = 0.5 * (1 + np.tanh((- grad + bias) / t))
        act[:] = next_act * learning_rate + act * (1. - learning_rate)


def should_stop(act: ndarray, acts: List[ndarray], min_act_change: float = 1e-5, lookback: int = 1) -> bool:
    return max(np.max(act - a0) for a0 in acts[-lookback:]) < min_act_change


def energy(matrix: Union[spmatrix, ndarray], act: ndarray):
    return matrix.dot(act).dot(act)


def energy_gradient(matrix: Union[spmatrix, ndarray], act: ndarray):
    return 2 * matrix.dot(act)


def build_segmented_tracks(hits):
    tracks = []
    for track, g in hits.groupby('track'):
        if track >= 0:
            segments = []
            for i in range(min(g.layer), max(g.layer)):
                for a in g[g.layer == i].index:
                    for b in g[g.layer == i + 1].index:
                        segments.append((a, b))
            tracks.append((track, segments))
    return dict(tracks)


def enumerate_segmented_track(track: List[Tuple[int, int]], seg: np.array) -> List[int]:
    nom_seg=[]
    for step in track:
        for si, s in enumerate(seg):
            if (s[0] == step[0]) and (s[1] == step[1]):
                nom_seg.append(si)  
    return nom_seg


def found_tracks(seg: np.ndarray, act: np.ndarray, all_tracks: List[List[Tuple[int, int]]]) -> int:
    
    kol_act_track=0
    all_kol_track=0
    end_track=False
    i=0
    lseg=[]
    for ti, t in (enumerate(all_tracks)):
        kol_act_seg=0
        all_kol_track+=1
        Track_seg=enumerate_segmented_track(t,seg)
        for s in Track_seg:
            if (act[s] >= 0.5):
                kol_act_seg+=1
            
        if kol_act_seg==len(Track_seg):    
            kol_act_track+=1
    return kol_act_track


def found_crosses(seg: np.ndarray, act: np.ndarray) -> int:
    kol_crosses=0
    
    for si, s in enumerate(seg):
        if (act[si] >= 0.5):

            for si_tek,s_tek in enumerate(seg):
                if (act[si_tek] >= 0.5):
                    if s_tek[0]  == s[0] and s_tek[1] != s[1]:
                        kol_crosses+=1
                        
                if (act[si_tek] >= 0.5):
                    if s_tek[0]  != s[0] and s_tek[1] == s[1]:
                        kol_crosses+=1
    return(kol_crosses)
