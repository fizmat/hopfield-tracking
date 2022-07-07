import numpy as np
import pandas as pd

from tracking.segment import gen_seg_track_layered


def gen_perfect_act(hits: pd.DataFrame, seg: np.ndarray) -> np.ndarray:
    perfect_act = np.zeros(len(seg))
    track_segment_set = set(tuple(s) for s in gen_seg_track_layered(hits))
    is_in_track = np.array([tuple(s) in track_segment_set for s in seg])
    if len(is_in_track):
        perfect_act[is_in_track] = 1
    return perfect_act
