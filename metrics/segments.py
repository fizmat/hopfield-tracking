import numpy as np
import pandas as pd


def gen_perfect_act(seg: np.ndarray, tseg: np.ndarray) -> np.ndarray:
    # Can't just use hits.track[a] = hits.track[b]: if there's a segment connecting far away hits in the track,
    # they should not activate.
    dfseg = pd.DataFrame(seg, columns=('a', 'b'))
    dftseg = pd.DataFrame(tseg, columns=('a', 'b'))
    df = pd.merge(dfseg, dftseg, on=['a', 'b'], how='left', indicator=True)
    return (df._merge == 'both').to_numpy().astype(float)


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
