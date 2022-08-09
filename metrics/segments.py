import numpy as np


def gen_perfect_act(seg: np.ndarray, tseg: np.ndarray) -> np.ndarray:
    # Can't just use hits.track[a] = hits.track[b]: if there's a segment connecting far away hits in the track,
    # they should not activate.
    perfect_act = np.zeros(len(seg))
    track_segment_set = set(tuple(s) for s in tseg)
    is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
    if len(is_in_track):
        perfect_act[is_in_track] = 1
    return perfect_act


def _profile():
    from datasets import get_hits
    from segment.candidate import gen_seg_layered
    from segment.track import gen_seg_track_layered
    event = get_hits('trackml_volume', 1)
    seg = gen_seg_layered(event)
    tseg = gen_seg_track_layered(event)
    gen_perfect_act(seg, tseg)


if __name__ == '__main__':
    _profile()
