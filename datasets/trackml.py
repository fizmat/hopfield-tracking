import math
from pathlib import Path
from typing import Optional, Generator
from zipfile import ZipFile

import dask.dataframe as dd
import pandas as pd
from dask import delayed
from numpy.testing import assert_array_equal
from tqdm.dask import TqdmCallback
from trackml.dataset import load_dataset, load_event

LAYER_DIST = 1e4  # actually peaks at 20 and 1e4-1e5
PATH = Path(__file__).parents[1] / 'data' / 'trackml'
EVENT_PREFIX = (PATH / f'event000001000').resolve()
SAMPLE_ZIP = PATH / 'train_sample.zip'
TRAIN1_ZIP = PATH / 'train_1.zip'
BLACKLIST_ZIP = PATH / 'blacklist_training.zip'
EVENT_FEATHER = PATH / 'event000001000.feather'
HITS_FEATHER = PATH / 'train_sample.feather'
PARTICLES_FEATHER = PATH / 'train_sample.particles.feather'
CELLS_FEATHER = PATH / 'train_sample.cells.feather'
HITS_PARQUET = PATH / 'train_1' / 'hits'
PARTICLES_PARQUET = PATH / 'train_1' / 'particles'
CELLS_PARQUET = PATH / 'train_1' / 'cells'


def _blacklist_hits(hits, blacklist_hits, blacklist_particles):
    hits['blacklisted'] = hits.hit_id.isin(set(blacklist_hits.hit_id))
    assert_array_equal(hits.blacklisted, hits.particle_id.isin(set(blacklist_particles.particle_id)))
    return hits


def _transform(hits):
    hits.rename(columns={'layer_id': 'layer', 'particle_id': 'track'}, inplace=True)
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits['layer'] = hits.layer // 2
    return hits


def _zip_sample_generator(n_events: Optional[int] = None, path=TRAIN1_ZIP, skip: Optional[int] = None,
                          ) -> Generator[pd.DataFrame, None, None]:
    with ZipFile(BLACKLIST_ZIP) as bz:
        for event_id, hits, truth in load_dataset(path, nevents=n_events, skip=skip, parts=['hits', 'truth']):
            hits = hits.set_index('hit_id').join(truth.set_index('hit_id')).reset_index()
            with bz.open(f'event{event_id:09}-blacklist_hits.csv') as f:
                blacklist_hits = pd.read_csv(f)
            with bz.open(f'event{event_id:09}-blacklist_particles.csv') as f:
                blacklist_particles = pd.read_csv(f)
            hits = _blacklist_hits(hits, blacklist_hits, blacklist_particles)
            hits.insert(0, 'event_id', event_id)
            yield hits


def _zip_extra_generator(n_events: Optional[int] = None, path=TRAIN1_ZIP, skip: Optional[int] = None,
                         parts: str = 'particles') -> Generator[pd.DataFrame, None, None]:
    for event_id, df in load_dataset(path, nevents=n_events, skip=skip, parts=[parts]):
        df.insert(0, 'event_id', event_id)
        yield df


def _zip_sample(n_events: Optional[int] = None, path: Path = SAMPLE_ZIP,
                skip: Optional[int] = None) -> pd.DataFrame:
    return pd.concat(list(_zip_sample_generator(n_events, path, skip)), ignore_index=True)


def _zip_extra(n_events: Optional[int] = None, path: Path = SAMPLE_ZIP,
               skip: Optional[int] = None, parts: str = 'particles') -> pd.DataFrame:
    events = list(_zip_extra_generator(n_events, path, skip, parts))
    return pd.concat(events, ignore_index=True) if events else None


def _zip_hits_dask(n_events: int = 1770, path=TRAIN1_ZIP, batch_size=50) -> dd.DataFrame:
    return dd.from_delayed([delayed(_zip_sample)(batch_size, path, cursor)
                            for cursor in range(0, n_events, batch_size)])


def _zip_extra_dask(n_events: int = 1770, path=TRAIN1_ZIP, batch_size=50, parts: str = 'particles') -> dd.DataFrame:
    return dd.from_delayed([delayed(_zip_extra)(batch_size, path, cursor, parts)
                            for cursor in range(0, n_events, batch_size)])


def _csv_one_event():
    hits, truth = load_event(EVENT_PREFIX, ['hits', 'truth'])
    hits = hits.set_index('hit_id').join(truth.set_index('hit_id')).reset_index()
    blacklist_hits = pd.read_csv(f'{EVENT_PREFIX}-blacklist_hits.csv')
    blacklist_particles = pd.read_csv(f'{EVENT_PREFIX}-blacklist_particles.csv')
    hits = _blacklist_hits(hits, blacklist_hits, blacklist_particles)
    hits.insert(0, 'event_id', 1000)
    return hits


def _feather_one_event() -> pd.DataFrame:
    return pd.read_feather(EVENT_FEATHER)


def _feather_sample(n_events: Optional[int] = None) -> pd.DataFrame:
    hits = pd.read_feather(HITS_FEATHER)
    if n_events is None:
        return hits
    return hits[hits.event_id.isin(hits.event_id.unique()[:n_events])]


def get_one_event() -> pd.DataFrame:
    event = _feather_one_event() if EVENT_FEATHER.exists() else _csv_one_event()
    return _transform(event)


def get_sample(n_events: Optional[int] = None) -> pd.DataFrame:
    sample = _feather_sample(n_events) if HITS_FEATHER.exists() else _zip_sample(n_events, SAMPLE_ZIP)
    return _transform(sample)


def gen_train_1(n_events: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
    yield from map(_transform, _zip_sample_generator(n_events))


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
    _csv_one_event().to_feather(EVENT_FEATHER, compression='zstd', compression_level=18)
    _zip_sample().to_feather(HITS_FEATHER, compression='zstd', compression_level=18)
    _zip_extra().to_feather(PARTICLES_FEATHER, compression='zstd', compression_level=18)
    _zip_extra(parts='cells').to_feather(CELLS_FEATHER, compression='zstd', compression_level=18)


def _create_parquets() -> None:
    with TqdmCallback(desc="hits"):
        _zip_hits_dask(batch_size=50).to_parquet(
            HITS_PARQUET, overwrite=True, write_metadata_file=True, compression='zstd', compression_level=18)
    with TqdmCallback(desc="particles"):
        _zip_extra_dask(batch_size=400).to_parquet(
            PARTICLES_PARQUET, overwrite=True, write_metadata_file=True, compression='zstd', compression_level=18)
    with TqdmCallback(desc="cells"):
        _zip_extra_dask(batch_size=20, parts='cells').to_parquet(
            CELLS_PARQUET, overwrite=True, write_metadata_file=True, compression='zstd', compression_level=18)


def main():
    _create_feathers()
    _create_parquets()


if __name__ == '__main__':
    main()
