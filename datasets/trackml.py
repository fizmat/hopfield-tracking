from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pandas as pd
from trackml.dataset import load_dataset, load_event

LAYER_DIST = 1e4  # actually peaks at 20 and 1e4-1e5
PATH = Path(__file__).parents[1] / 'data' / 'trackml'
EVENT_PREFIX = (PATH / f'event000001000').resolve()
SAMPLE_ZIP = PATH / 'train_sample.zip'
TRAIN1_ZIP = PATH / 'train_1.zip'
BLACKLIST_ZIP = PATH / 'blacklist_training.zip'


def _transform(hits, blacklist_hits):
    hits = hits.drop(index=blacklist_hits.index, errors='ignore')
    hits.rename(columns={'layer_id': 'layer', 'particle_id': 'track'}, inplace=True)
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits['layer'] = hits.layer // 2
    return hits


def get_hits_trackml(n_events: Optional[int] = None, path=SAMPLE_ZIP) -> pd.DataFrame:
    events = []
    with ZipFile(BLACKLIST_ZIP) as bz:
        for event_id, hits, truth in load_dataset(path, nevents=n_events, parts=['hits', 'truth']):
            hits.set_index('hit_id', inplace=True)
            truth.set_index('hit_id', inplace=True)
            hits = hits.join(truth)
            with bz.open(f'event{event_id:09}-blacklist_hits.csv') as f:
                blacklist_hits = pd.read_csv(f, index_col='hit_id')
            hits = _transform(hits, blacklist_hits)
            hits['event_id'] = event_id
            events.append(hits)
    return pd.concat(events, ignore_index=True)


def get_hits_trackml_one_event():
    hits, truth = load_event(EVENT_PREFIX, ['hits', 'truth'])
    hits.set_index('hit_id', inplace=True)
    truth.set_index('hit_id', inplace=True)
    hits = hits.join(truth)
    blacklist_hits = pd.read_csv(f'{EVENT_PREFIX}-blacklist_hits.csv', index_col='hit_id')
    hits = _transform(hits, blacklist_hits)
    hits['event_id'] = 1000
    return hits


def get_hits_trackml_by_volume(n_events: Optional[int] = None, path=SAMPLE_ZIP) -> pd.DataFrame:
    hits = get_hits_trackml(n_events=n_events, path=path)
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits if n_events is None else hits[hits.event_id.isin(hits.event_id.unique()[:n_events])]


def get_hits_trackml_one_event_by_volume():
    hits = get_hits_trackml_one_event()
    hits = hits[hits.volume_id == 7]
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits


def main():
    df = get_hits_trackml()
    df.info()


if __name__ == '__main__':
    main()
