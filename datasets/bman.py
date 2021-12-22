def _read():
    import pandas as pd
    simdata = pd.read_csv('../data/bman/simdata_ArPb_3.2AGeV_mb_1.zip', sep='\t',
                          names=['event_id', 'x', 'y', 'z', 'detector_id', 'station_id', 'track_id', 'px', 'py', 'pz',
                                 'vx', 'vy', 'vz'])
    return simdata


def _transform(simdata, max_hits=None):
    if max_hits is not None:
        hit_count = simdata.groupby('event_id').size()
        small_events = set(hit_count[hit_count <= max_hits].index)
        simdata = simdata[simdata.event_id.isin(small_events)].copy()
    simdata['layer'] = simdata.detector_id * 3 + simdata.station_id
    return simdata.rename(columns={'track_id': 'track'})[['x', 'y', 'z', 'layer', 'track', 'event_id']]


def get_hits_bman(max_hits=None):
    return _transform(_read(), max_hits)
