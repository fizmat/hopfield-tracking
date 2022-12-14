from typing import List, Tuple, Hashable, Dict, Iterable

import numpy as np
import pandas as pd

from metrics.segments import gen_perfect_act


def build_segmented_tracks(hits: pd.DataFrame) -> Dict[Hashable, List[Tuple[int, int]]]:
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
    nom_seg = []
    for step in track:
        z = np.zeros_like(seg)
        condition = ((seg[:, 0] == step[0]) & (seg[:, 1] == step[1]))
        tek_nom_seg = np.arange(len(seg))[condition][0]
        nom_seg.append(tek_nom_seg)
    return nom_seg


def found_tracks(seg: np.ndarray, act: np.ndarray, all_tracks: Iterable[List[Tuple[int, int]]]) -> int:
    kol_act_track = 0
    all_kol_track = 0
    end_track = False
    i = 0
    lseg = []
    for ti, t in (enumerate(all_tracks)):
        kol_act_seg = 0
        all_kol_track += 1
        Track_seg = enumerate_segmented_track(t, seg)
        for s in Track_seg:
            if (act[s] >= 0.5):
                kol_act_seg += 1

        if kol_act_seg == len(Track_seg):
            kol_act_track += 1
    return kol_act_track


def found_crosses(seg: np.ndarray, act: np.ndarray) -> int:
    kol_crosses = 0
    seg_tek = seg[act >= 0.5]
    for si, s in enumerate(seg):
        if (act[si] >= 0.5):
            condition1 = (seg_tek[:, 0] == s[0]) & (seg_tek[:, 1] != s[1])
            kol_crosses += np.sum(condition1)
            condition2 = (seg_tek[:, 0] != s[0]) & (seg_tek[:, 1] == s[1])
            kol_crosses += np.sum(condition2)
    return (kol_crosses // 2)


def track_metrics(hits: pd.DataFrame, seg: np.ndarray, act: np.ndarray, threshold: float) -> Dict[str, int]:
    perfect_act = gen_perfect_act(hits, seg)
    reds = np.sum((act > threshold) & (perfect_act < threshold))
    segmented_tracks = build_segmented_tracks(hits).values()
    tracks = found_tracks(seg, act, segmented_tracks)
    crosses = found_crosses(seg, act)
    return {'reds': reds, 'tracks': tracks, 'crosses': crosses}


def track_loss(metrics: pd.DataFrame) -> pd.Series:
    return -(metrics.tracks - metrics.crosses - 0.036 * metrics.reds)
