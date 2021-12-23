from pathlib import Path

import numpy as np
import pandas as pd


def _read_truth(event_file: str, prefix: str = None) -> pd.DataFrame:
    if prefix is None:
        prefix = Path(__file__) / '../../data/trackml/train_100_events'
    else:
        prefix = Path(prefix)
    hits = pd.read_csv(prefix / (event_file + '-hits.csv'))
    truth = pd.read_csv(prefix / (event_file + '-truth.csv'))
    return hits.merge(truth, on='hit_id')


def _read_blacklist(event_file: str, prefix: str = None) -> pd.DataFrame:
    if prefix is None:
        prefix = Path(__file__) / '../../data/trackml/blacklist'
    else:
        prefix = Path(prefix)
    return pd.read_csv(prefix / (event_file + '-blacklist_hits.csv'))


def _transform(hits_truth, blacklist_hits):
    hits_trackML = hits_truth[np.logical_not(hits_truth.hit_id.isin(blacklist_hits.hit_id))]
    hits = hits_trackML.rename(columns={'layer_id': 'layer', 'particle_id': 'track'})
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits.reset_index(drop=True, inplace=True)
    hits['layer'] = hits.layer // 2
    return hits


def get_hits_trackml():
    events = []
    event_prefix = 'event00000'

    for num_ev in range(1000, 1100):
        event_file = event_prefix + f'{num_ev:04}'
        hits_truth = _read_truth(event_file)
        blacklist = _read_blacklist(event_file)
        hits = _transform(hits_truth, blacklist)
        hits['event_id'] = num_ev
        events.append(hits)
    return pd.concat(events, ignore_index=True)


def get_hits_trackml_by_volume():
    hits = get_hits_trackml()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits


def get_hits_trackml_by_module():
    hits = get_hits_trackml()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str) + '-' + hits.module_id.astype(str)
    return hits
