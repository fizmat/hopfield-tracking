from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def _transform(hits, blacklist_hits):
    hits = hits[np.logical_not(hits.hit_id.isin(blacklist_hits.hit_id))]
    hits = hits.rename(columns={'layer_id': 'layer', 'particle_id': 'track'})
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits.reset_index(drop=True, inplace=True)
    hits['layer'] = hits.layer // 2
    return hits


def get_hits_trackml(zip: Path = Path(__file__).parents[1] / 'data/trackml/train_sample.zip',
                     blacklist_zip: Path = Path(__file__).parents[1] / 'data/trackml/blacklist_training.zip') \
        -> pd.DataFrame:
    events = []
    with ZipFile(zip) as z:
        with ZipFile(blacklist_zip) as bz:
            for event_number in range(1000, 1100):
                with z.open(f'train_100_events/event{event_number:09}-hits.csv') as f:
                    hits = pd.read_csv(f)
                with z.open(f'train_100_events/event{event_number:09}-truth.csv') as f:
                    truth = pd.read_csv(f)
                hits = hits.merge(truth, on='hit_id')
                with bz.open(f'event{event_number:09}-blacklist_hits.csv') as f:
                    blacklist_hits = pd.read_csv(f)
                hits = _transform(hits, blacklist_hits)
                hits['event_id'] = event_number
                events.append(hits)
    return pd.concat(events, ignore_index=True)


def get_hits_trackml_one_event(path: Path = Path(__file__).parents[1] / 'data/trackml'):
    event_number = 1000
    hits = pd.read_csv(path / f'event{event_number:09}-hits.csv')
    truth = pd.read_csv(path / f'event{event_number:09}-truth.csv')
    hits = hits.merge(truth, on='hit_id')
    blacklist_hits = pd.read_csv(path / f'event{event_number:09}-blacklist_hits.csv')
    hits = _transform(hits, blacklist_hits)
    hits['event_id'] = 1000
    return hits


def get_hits_trackml_by_volume():
    hits = get_hits_trackml()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits


def get_hits_trackml_one_event_by_volume():
    hits = get_hits_trackml_one_event()
    hits = hits[hits.volume_id == 7].reset_index(drop=True)
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits


def get_hits_trackml_by_module():
    hits = get_hits_trackml()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str) + '-' + hits.module_id.astype(str)
    return hits


def get_hits_trackml_one_event_by_module():
    hits = get_hits_trackml_one_event()
    hits = hits[np.logical_and(hits.volume_id == 7, hits.module_id == 1)].reset_index(drop=True)
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str) + '-' + hits.module_id.astype(str)
    return hits
