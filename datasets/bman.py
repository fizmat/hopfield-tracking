from pathlib import Path

import pandas as pd


def _read(prefix: str = None, file: str = 'simdata_ArPb_3.2AGeV_mb_1.zip') -> pd.DataFrame:
    if prefix is None:
        prefix = Path(__file__).parents[1] / 'data/bman'
    file = Path(prefix) / file
    simdata = pd.read_csv(file, sep='\t',
                          names=['event_id', 'x', 'y', 'z', 'detector_id', 'station_id', 'track_id',
                                 'px', 'py', 'pz', 'vx', 'vy', 'vz'])
    return simdata


def _transform(simdata, max_hits=None):
    if max_hits is not None:
        hit_count = simdata.groupby('event_id').size()
        small_events = set(hit_count[hit_count <= max_hits].index)
        simdata = simdata[simdata.event_id.isin(small_events)].copy()
    simdata['layer'] = simdata.detector_id * 3 + simdata.station_id
    return simdata.rename(columns={'track_id': 'track'})[['event_id', 'x', 'y', 'z', 'layer', 'track']]


def _copy_hits_bman_event6():
    events = _read()
    e6 = events[events.event_id == 6]
    e6.to_csv(Path(__file__).parents[1] / 'data/bman/event6.csv',
              sep='\t', header=False, index=False)


def get_hits_bman(max_hits=None):
    return _transform(_read(), max_hits)


def get_hits_bman_one_event():
    return _transform(_read(file='event6.csv'), None)
