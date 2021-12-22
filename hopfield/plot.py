from typing import Tuple, List, Dict, Any

from numpy import ndarray


def make_tracks_3d(
        pos: ndarray, seg: ndarray, act: ndarray, perfect_act: ndarray, threshold: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    segment_paths = [
        {'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
         'act': a, 'perfect_act': pa,
         'positive': a > threshold,
         'true': pa > threshold,
         } for xyz, a, pa in zip(pos[seg], act, perfect_act)
    ]
    tp = [p for p in segment_paths if p['true'] and p['positive']]
    fp = [p for p in segment_paths if not p['true'] and p['positive']]
    tn = [p for p in segment_paths if not p['true'] and not p['positive']]
    fn = [p for p in segment_paths if p['true'] and not p['positive']]
    return tp, fp, tn, fn