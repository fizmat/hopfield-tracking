import numpy as np
import pandas as pd


def gen_perfect_act(seg: np.ndarray, tseg: np.ndarray) -> np.ndarray:
    # Can't just use hits.track[a] = hits.track[b]: if there's a segment connecting far away hits in the track,
    # they should not activate.
    perfect_act = np.zeros(len(seg))
    dfseg = pd.MultiIndex.from_arrays([seg[:, 0], seg[:, 1]])
    dftseg = pd.MultiIndex.from_arrays([tseg[:, 0], tseg[:, 1]])
    _, indexer, __ = dfseg.join(dftseg, how='inner', return_indexers=True)
    perfect_act[indexer] = 1.
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
