import numpy as np

def input_hits_TrackML(event_file):
    import pandas as pd
      
    hits_truth=[]
    file_hits = 'datasets/trackml/train_100_events/'+event_file+'-hits.csv'
    file_truth = 'datasets/trackml/train_100_events/'+event_file+'-truth.csv'           
    hits = pd.read_csv(file_hits)
    truth = pd.read_csv(file_truth)
    hits_truth = hits.merge(truth, left_on='hit_id', right_on='hit_id', suffixes=('_left', '_right'))
    return hits_truth

def input_blacklist_TrackML(event_file):
    import pandas as pd
    
    file_blacklist_hits = 'datasets/trackml/blacklist/'+event_file+'-blacklist_hits.csv'
    file_blacklist_particles = 'datasets/trackml/blacklist/'+event_file+'-blacklist_particles.csv'
    blacklist_hits = pd.read_csv(file_blacklist_hits)
    blacklist_particle = pd.read_csv(file_blacklist_particles)
    return blacklist_hits


def transform_TrackML(hits_truth, blacklist_hits):
    hits_trackML=hits_truth[np.logical_not(hits_truth.hit_id.isin(blacklist_hits.hit_id))]

    hits=hits_trackML.rename(columns={'layer_id': 'layer', 'particle_id': 'track'})
    hits.track = hits.track.where(hits.track!=0, other=-1)
    hits = hits[np.logical_and(hits.volume_id==7, hits.module_id==1)]
    hits.reset_index(drop=True, inplace=True)
    hits['layer'] = hits.layer//2
    return hits 


def input_BMaN():
    import pandas as pd
    simdata = pd.read_csv('simdata_ArPb_3.2AGeV_mb_1.zip', sep='\t',
                          names=['event_id', 'x', 'y', 'z', 'detector_id', 'station_id', 'track_id', 'px', 'py', 'pz', 'vx', 'vy', 'vz'])
    return(simdata)


def transform_BMaN(simdata):
    cc = simdata.groupby('event_id').x.count()
    cc[cc==cc[cc<=800].max()]
    simdata['layer'] = simdata.detector_id * 3 + simdata.station_id
    return simdata.rename(columns={'track_id': 'track'})[['x', 'y', 'z', 'layer', 'track', 'event_id']]
