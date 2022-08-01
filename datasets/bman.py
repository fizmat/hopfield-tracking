from pathlib import Path
from typing import Optional
from zipfile import ZipFile, BadZipFile

import pandas as pd


def _read(prefix: str = None, file: str = 'simdata_ArPb_3.2AGeV_mb_1.zip') -> pd.DataFrame:
    if prefix is None:
        prefix = Path(__file__).parents[1] / 'data/bman'
    file = Path(prefix) / file
    col_names = ['event_id', 'x', 'y', 'z', 'detector', 'station', 'track', 'px', 'py', 'pz', 'vx', 'vy', 'vz']
    try:
        with ZipFile(file) as z:
            with z.open(f'simdata_ArPb_3.2AGeV_mb_1.txt') as f:
                return pd.read_csv(f, sep='\t', names=col_names)
    except BadZipFile:
        return pd.read_csv(file, sep='\t', names=col_names)


def _transform(simdata, max_hits=None):
    if max_hits is not None:
        hit_count = simdata.groupby('event').size()
        small_events = set(hit_count[hit_count <= max_hits].index)
        simdata = simdata[simdata.event_id.isin(small_events)].copy()
    simdata['layer'] = simdata.detector * 3 + simdata.station
    return simdata[['event_id', 'x', 'y', 'z', 'layer', 'track']]


def _copy_hits_bman_event6():
    hits = _read()
    e6 = hits[hits.event_id == 6]
    e6.to_csv(Path(__file__).parents[1] / 'data/bman/event6.csv',
              sep='\t', header=False, index=False)


def get_hits_bman(n_events: Optional[int] = None, max_hits: Optional[int] = None) -> pd.DataFrame:
    hits = _transform(_read(), max_hits)
    return hits if n_events is None else hits[hits.event_id.isin(hits.event_id.unique()[:n_events])]


def get_hits_bman_one_event():
    return _transform(_read(file='event6.csv'), None)
