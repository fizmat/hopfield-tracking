from pathlib import Path
from typing import Optional

import pandas as pd
from numpy import single, short, byte, double

LAYER_DIST = 20.
PATH = Path(__file__).parents[1] / 'data' / 'bman'
FILE_NAME = 'simdata_ArPb_3.2AGeV_mb_1'
ZIP_FILE = PATH / f'{FILE_NAME}.zip'
FEATHER_FILE = PATH / f'{FILE_NAME}.feather'
CSV_EVENT = PATH / 'event6.csv'
SCHEMA = {
    'event_id': short,
    'x': single, 'y': single, 'z': single,
    'detector': byte, 'station': byte, 'track': short,
    'px': double, 'py': double, 'pz': double,
    'vx': double, 'vy': double, 'vz': double
}
KEEP_COLUMNS = ['event_id', 'x', 'y', 'z', 'layer', 'track']


def _read_zip() -> pd.DataFrame:
    df = pd.read_csv(ZIP_FILE, sep='\t', names=list(SCHEMA.keys()), dtype=SCHEMA)
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
    return pd.read_csv(CSV_EVENT, dtype=SCHEMA)


def main():
    hits = _read_zip()
    hits[hits.event_id == 6].to_csv(CSV_EVENT, index=False)
    hits.to_feather(FEATHER_FILE, compression='zstd', compression_level=18)


if __name__ == '__main__':
    main()
