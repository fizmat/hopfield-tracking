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


def _read_truth_TrackML(event_file):
    hits_truth = []
    file_hits = 'datasets/trackml/train_100_events/' + event_file + '-hits.csv'
    file_truth = 'datasets/trackml/train_100_events/' + event_file + '-truth.csv'
    hits = pd.read_csv(file_hits)
    truth = pd.read_csv(file_truth)
    hits_truth = hits.merge(truth, left_on='hit_id', right_on='hit_id', suffixes=('_left', '_right'))
    return hits_truth


def _read_blacklist_TrackML(event_file):
    file_blacklist_hits = 'datasets/trackml/blacklist/' + event_file + '-blacklist_hits.csv'
    # file_blacklist_particles = 'datasets/trackml/blacklist/'+event_file+'-blacklist_particles.csv'
    blacklist_hits = pd.read_csv(file_blacklist_hits)
    # blacklist_particles = pd.read_csv(file_blacklist_particles)
    return blacklist_hits


def _transform_TrackML(hits_truth, blacklist_hits):
    hits_trackML = hits_truth[np.logical_not(hits_truth.hit_id.isin(blacklist_hits.hit_id))]

    hits = hits_trackML.rename(columns={'layer_id': 'layer', 'particle_id': 'track'})
    hits.track = hits.track.where(hits.track != 0, other=-1)
    hits.reset_index(drop=True, inplace=True)
    hits['layer'] = hits.layer // 2
    return hits


def get_hits_TrackML():
    events = []
    event_prefix = 'event00000'

    for num_ev in range(1000, 1100):
        strnum_ev = f'{num_ev:04}'

        event_file = event_prefix + strnum_ev
        hits_truth = _read_truth_TrackML(event_file)
        blacklist = _read_blacklist_TrackML(event_file)
        hits = _transform_TrackML(hits_truth, blacklist)
        hits['event_id'] = num_ev
        events.append(hits)
    return pd.concat(events, ignore_index=True)


def get_hits_TrackML_by_volume():
    hits = get_hits_TrackML()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str)
    return hits


def get_hits_TrackML_by_module():
    hits = get_hits_TrackML()
    hits.event_id = hits.event_id.astype(str) + '-' + hits.volume_id.astype(str) + '-' + hits.module_id.astype(str)
    return hits

def _read_BMaN():
    import pandas as pd
    simdata = pd.read_csv('simdata_ArPb_3.2AGeV_mb_1.zip', sep='\t',
                          names=['event_id', 'x', 'y', 'z', 'detector_id', 'station_id', 'track_id', 'px', 'py', 'pz',
                                 'vx', 'vy', 'vz'])
    return (simdata)


def _transform_BMaN(simdata, max_hits=400):
    hit_count = simdata.groupby('event_id').size()
    small_events = set(hit_count[hit_count <= max_hits].index)
    simdata = simdata[simdata.event_id.isin(small_events)].copy()
    simdata['layer'] = simdata.detector_id * 3 + simdata.station_id
    return simdata.rename(columns={'track_id': 'track'})[['x', 'y', 'z', 'layer', 'track', 'event_id']]


def get_hits_BMaN(max_hits=400):
    return _transform_BMaN(_read_BMaN(), max_hits)
