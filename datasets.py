import numpy as np
import pandas as pd

from generator import SimpleEventGenerator


def get_hits_simple(n_events=100, event_size=10):
    hits_list = []
    for i, event in enumerate(SimpleEventGenerator().gen_many_events(n_events, event_size)):
        hits, seg = event
        hits['event_id'] = i
        hits_list.append(hits)
    return pd.concat(hits_list, ignore_index=True)


def _read_truth_trackml(event_file):
    hits = pd.read_csv('data/trackml/train_100_events/' + event_file + '-hits.csv')
    truth = pd.read_csv('data/trackml/train_100_events/' + event_file + '-truth.csv')
    return hits.merge(truth, on='hit_id')


def _read_blacklist_trackml(event_file):
    return pd.read_csv('data/trackml/blacklist/' + event_file + '-blacklist_hits.csv')


def _transform_trackml(hits_truth, blacklist_hits):
    hits_trackML = hits_truth[np.logical_not(hits_truth.hit_id.isin(blacklist_hits.hit_id))]
    hits = hits_trackML.rename(columns={'layer_id': 'layer', 'particle_id': 'track'})
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits.reset_index(drop=True, inplace=True)
    hits['layer'] = hits.layer // 2
    return hits


def get_hits_trackml():
    events = []
    event_prefix = 'event00000'

    for num_ev in range(1000, 1100):
        event_file = event_prefix + f'{num_ev:04}'
        hits_truth = _read_truth_trackml(event_file)
        blacklist = _read_blacklist_trackml(event_file)
        hits = _transform_trackml(hits_truth, blacklist)
        hits['event_id'] = num_ev
        events.append(hits)
    return pd.concat(events, ignore_index=True)


def get_hits_trackml_by_volume():
    hits = get_hits_trackml()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits


def get_hits_trackml_by_module():
    hits = get_hits_trackml()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str) + '-' + hits.module_id.astype(str)
    return hits


def _read_bman():
    import pandas as pd
    simdata = pd.read_csv('data/bman/simdata_ArPb_3.2AGeV_mb_1.zip', sep='\t',
                          names=['event_id', 'x', 'y', 'z', 'detector_id', 'station_id', 'track_id', 'px', 'py', 'pz',
                                 'vx', 'vy', 'vz'])
    return simdata


def _transform_bman(simdata, max_hits=None):
    if max_hits is not None:
        hit_count = simdata.groupby('event_id').size()
        small_events = set(hit_count[hit_count <= max_hits].index)
        simdata = simdata[simdata.event_id.isin(small_events)].copy()
    simdata['layer'] = simdata.detector_id * 3 + simdata.station_id
    return simdata.rename(columns={'track_id': 'track'})[['x', 'y', 'z', 'layer', 'track', 'event_id']]


def get_hits_bman(max_hits=None):
    return _transform_bman(_read_bman(), max_hits)
