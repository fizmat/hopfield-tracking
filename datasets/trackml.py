import math
from pathlib import Path
from typing import Optional, Generator
from zipfile import ZipFile

import pandas as pd
from trackml.dataset import load_dataset, load_event

LAYER_DIST = 1e4  # actually peaks at 20 and 1e4-1e5
PATH = Path(__file__).parents[1] / 'data' / 'trackml'
EVENT_PREFIX = (PATH / f'event000001000').resolve()
SAMPLE_ZIP = PATH / 'train_sample.zip'
TRAIN1_ZIP = PATH / 'train_1.zip'
TRAIN1_DIR = PATH / 'train_1'
BLACKLIST_ZIP = PATH / 'blacklist_training.zip'
EVENT_FEATHER = PATH / 'event000001000.feather'
SAMPLE_FEATHER = PATH / 'train_sample.feather'


def _transform(hits, blacklist_hits):
    hits = hits[~hits.hit_id.isin(set(blacklist_hits.hit_id))].reset_index(drop=True)
    hits.rename(columns={'layer_id': 'layer', 'particle_id': 'track'}, inplace=True)
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits['layer'] = hits.layer // 2
    return hits


def _zip_sample_generator(n_events: Optional[int] = None, path=TRAIN1_ZIP) -> Generator[pd.DataFrame, None, None]:
    with ZipFile(BLACKLIST_ZIP) as bz:
        for event_id, hits, truth in load_dataset(path, nevents=n_events, parts=['hits', 'truth']):
            hits = hits.set_index('hit_id').join(truth.set_index('hit_id')).reset_index()
            with bz.open(f'event{event_id:09}-blacklist_hits.csv') as f:
                blacklist_hits = pd.read_csv(f)
            hits = _transform(hits, blacklist_hits)
            hits['event_id'] = event_id
            yield hits


def _zip_sample(n_events: Optional[int] = None, path=SAMPLE_ZIP) -> pd.DataFrame:
    return pd.concat(list(_zip_sample_generator(n_events, path)), ignore_index=True)


def _csv_one_event():
    hits, truth = load_event(EVENT_PREFIX, ['hits', 'truth'])
    hits = hits.set_index('hit_id').join(truth.set_index('hit_id')).reset_index()
    blacklist_hits = pd.read_csv(f'{EVENT_PREFIX}-blacklist_hits.csv')
    hits = _transform(hits, blacklist_hits).reset_index()
    hits['event_id'] = 1000
    return hits


def _feather_one_event() -> pd.DataFrame:
    return pd.read_feather(EVENT_FEATHER)


def _feather_sample(n_events: Optional[int] = None) -> pd.DataFrame:
    hits = pd.read_feather(SAMPLE_FEATHER)
    if n_events is None:
        return hits
    return hits[hits.event_id.isin(hits.event_id.unique()[:n_events])]


def get_one_event() -> pd.DataFrame:
    return _feather_one_event() if EVENT_FEATHER.exists() else _csv_one_event()


def get_sample(n_events: Optional[int] = None) -> pd.DataFrame:
    return _feather_sample(n_events) if SAMPLE_FEATHER.exists() else _zip_sample(n_events, SAMPLE_ZIP)


def get_train_1(n_events: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
    yield from _zip_sample_generator(n_events)


def get_sample_by_volume(n_events: Optional[int] = None) -> pd.DataFrame:
    to_read = None if n_events is None else math.ceil(n_events / 9)
    hits = get_sample(n_events=to_read)
    hits.event_id = hits.event_id * 100 + hits.volume_id
    if n_events is None:
        return hits
    return hits[hits.event_id.isin(hits.event_id.unique()[:n_events])]


def get_one_event_by_volume():
    hits = get_one_event()
    hits = hits[hits.volume_id == 7]
    hits.event_id = hits.event_id * 100 + hits.volume_id
    return hits


def _create_feathers():
    hits = _csv_one_event()
    hits.to_feather(EVENT_FEATHER, compression='zstd', compression_level=18)
    hits = _zip_sample()
    hits.to_feather(SAMPLE_FEATHER, compression='zstd', compression_level=18)


def _create_parquets():
    TRAIN1_DIR.mkdir(exist_ok=True)
    batch = eid = None
    for i, event in enumerate(_zip_sample_generator()):
        if batch is None:
            batch, eid = [], event.event_id.iloc[0]
        print(eid, len(batch))
        batch.append(event.set_index('event_id'))
        if len(batch) == 50:
            pd.concat(batch).to_parquet(TRAIN1_DIR / f'event{eid:09}.parquet', compression='brotli')
            batch = None
    if batch:
        pd.concat(batch).to_parquet(TRAIN1_DIR / f'event{eid:09}.parquet', compression='brotli')


def main():
    # _create_feathers()
    # _create_parquets()
    pass


if __name__ == '__main__':
    main()
