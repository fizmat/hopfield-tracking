import numpy as np


def gen_perfect_act(hits, seg):
    perfect_act = np.zeros(len(seg))
    track_segment_set = set(tuple(s) for s in mark_track_segments(hits))
    is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
    if len(is_in_track):
        perfect_act[is_in_track] = 1
    return perfect_act


def mark_track_segments(hits):
    track_segments = []
    for track, g in hits.groupby('track'):
        if track >= 0:
            for i in range(min(g.layer), max(g.layer)):
                for a in g[g.layer == i].index:
                    for b in g[g.layer == i + 1].index:
                        track_segments.append((a, b))
    return track_segments