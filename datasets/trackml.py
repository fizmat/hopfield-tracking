from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pandas as pd
from trackml.dataset import load_dataset, load_event

LAYER_DIST = 1e4  # actually peaks at 20 and 1e4-1e5


def _transform(hits, blacklist_hits):
    hits = hits.drop(index=blacklist_hits.index, errors='ignore')
    hits.rename(columns={'layer_id': 'layer', 'particle_id': 'track'}, inplace=True)
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits['layer'] = hits.layer // 2
    return hits


def get_hits_trackml(n_events: Optional[int] = None,
                     train_zip: Path = Path(__file__).parents[1] / 'data/trackml/train_sample.zip',
                     blacklist_zip: Path = Path(__file__).parents[1] / 'data/trackml/blacklist_training.zip',
                     ) -> pd.DataFrame:
    events = []
    with ZipFile(blacklist_zip) as bz:
        for event_id, hits, truth in load_dataset(train_zip.resolve(), nevents=n_events, parts=['hits', 'truth']):
            hits.set_index('hit_id', inplace=True)
            truth.set_index('hit_id', inplace=True)
            hits = hits.join(truth)
            with bz.open(f'event{event_id:09}-blacklist_hits.csv') as f:
                blacklist_hits = pd.read_csv(f, index_col='hit_id')
            hits = _transform(hits, blacklist_hits)
            hits['event_id'] = event_id
            events.append(hits)
    return pd.concat(events, ignore_index=True)


def get_hits_trackml_one_event(path: Path = Path(__file__).parents[1] / 'data/trackml'):
    event_number = 1000
    hits, truth = load_event(path / 'event000001000', ['hits', 'truth'])
    hits.set_index('hit_id', inplace=True)
    truth.set_index('hit_id', inplace=True)
    hits = hits.join(truth)
    blacklist_hits = pd.read_csv(path / f'event{event_number:09}-blacklist_hits.csv', index_col='hit_id')
    hits = _transform(hits, blacklist_hits)
    hits['event_id'] = 1000
    return hits


def get_hits_trackml_by_volume(n_events: Optional[int] = None, *args, **kwargs) -> pd.DataFrame:
    hits = get_hits_trackml(n_events=n_events, *args, **kwargs)
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits if n_events is None else hits[hits.event_id.isin(hits.event_id.unique()[:n_events])]


def get_hits_trackml_one_event_by_volume():
    hits = get_hits_trackml_one_event()
    hits = hits[hits.volume_id == 7]
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits
