from pathlib import Path
from typing import Optional

import pandas as pd

LAYER_DIST = 20.
PATH = Path(__file__).parents[1] / 'data' / 'bman'
FILE_NAME = 'simdata_ArPb_3.2AGeV_mb_1'
ZIP_FILE = PATH / f'{FILE_NAME}.zip'
FEATHER_FILE = PATH / f'{FILE_NAME}.feather'
CSV_EVENT = PATH / 'event6.csv'
COLUMN_NAMES = ['event_id', 'x', 'y', 'z', 'detector', 'station', 'track', 'px', 'py', 'pz', 'vx', 'vy', 'vz']
KEEP_COLUMNS = ['event_id', 'x', 'y', 'z', 'layer', 'track']


def _read_zip() -> pd.DataFrame:
    df = pd.read_csv(ZIP_FILE, sep='\t', names=COLUMN_NAMES)
    df['layer'] = df.detector * 3 + df.station
    return df[KEEP_COLUMNS]


def get_hits_bman(n_events: Optional[int] = None) -> pd.DataFrame:
    if FEATHER_FILE.exists():
        hits = pd.read_feather(FEATHER_FILE)
    else:
        hits = _read_zip()
    if n_events is not None:
        event_ids = hits.event_id.unique()[:n_events]
        hits = hits[hits.event_id.isin(event_ids)]
    return hits


def get_hits_bman_one_event():
    return pd.read_csv(CSV_EVENT)


def _copy_hits_bman_event6():
    hits = _read_zip()
    e6 = hits[hits.event_id == 6]
    e6.to_csv(CSV_EVENT, index=False)


def _copy_hits_bman_feather():
    hits = _read_zip()
    hits.to_feather(FEATHER_FILE, compression='zstd', compression_level=18)


if __name__ == '__main__':
    # _copy_hits_bman_event6()
    _copy_hits_bman_feather()
