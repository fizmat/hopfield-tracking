from typing import List, Tuple, Hashable, Dict, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from trackml.score import _analyze_tracks

from metrics.segments import gen_perfect_act
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_sequential


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


def track_metrics(event: pd.DataFrame, seg: np.ndarray, tseg: np.ndarray,
                  act: np.ndarray, positive: np.ndarray) -> Dict[str, float]:
    perfect_act = gen_perfect_act(seg, tseg)
    n_fp_seg = np.sum(positive & (perfect_act < 0.5))
    trackml, fakes, missed = trackml_score(event, seg, positive)
    return {
        'false positive segments': n_fp_seg,
        'trackml score': trackml,
        'false tracks': fakes,
        'missed tracks': missed
    }


def reconstruct_tracks(seg: np.ndarray, positive: np.ndarray) -> pd.DataFrame:
    g = nx.Graph([(a, b) for a, b in seg[positive]])
    df = pd.DataFrame([(hit_id, track_id)
                       for track_id, hit_indices in enumerate(nx.connected_components(g))
                       for hit_id in hit_indices], columns=('hit_id', 'track_id'))
    return df


def score_event(truth, submission):
    tracks = _analyze_tracks(truth, submission)
    # TODO: also zero score weight for one-hit track hits?
    tracks = tracks[np.logical_not(np.isnan(tracks.track_id))]
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
    return tracks['major_weight'][good_track].sum(), good_track.sum()


def trackml_score(event, seg, positive):
    reconstruction = reconstruct_tracks(seg, positive)
    truth = event.reset_index().rename(columns={'index': 'hit_id', 'track': 'particle_id'})
    if 'weight' not in truth:
        truth['weight'] = 1.
        truth.loc[truth.particle_id == -1, 'weight'] = 0
    n_tracks_total = reconstruction.groupby('track_id').count().hit_id.count()
    trackml, n_tracks_good = score_event(truth, reconstruction)
    fakes = n_tracks_total - n_tracks_good
    missed = len(set(truth.particle_id) - {-1}) - n_tracks_good
    return trackml, fakes, missed


def main():
    from datasets import get_hits
    event = get_hits('spdsim', 1)
    seg = gen_seg_layered(event)
    tseg = gen_seg_track_sequential(event)
    act = gen_perfect_act(seg, tseg)
    print(trackml_score(event, seg, act >= 0.5))


if __name__ == '__main__':
    main()
