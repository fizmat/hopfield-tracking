import numpy as np

def inputTrackML():
    import pandas as pd
      
    hits_truth=[]
    hits = pd.read_csv('event000001000-hits.csv')
    
    particle = pd.read_csv('event000001000-particles.csv')
    truth = pd.read_csv('event000001000-truth.csv')
    
    cels = pd.read_csv('event000001000-cells.csv')
    blacklist_hits = pd.read_csv('event000001000-blacklist_hits.csv')
    blacklist_particle = pd.read_csv('event000001000-blacklist_particles.csv')

    hits_truth = hits.merge(truth, left_on='hit_id', right_on='hit_id', suffixes=('_left', '_right'))
    
    #particle_truth = particle.merge(truth, left_on=’particle_id’,right_on=’particle_id’, suffixes=(‘_left’, ‘_right’))
    return hits_truth,blacklist_hits


def transformTrackML(hits_truth, blacklist_hits):
    hits_trackML=hits_truth[np.logical_not(hits_truth.hit_id.isin(blacklist_hits.hit_id))]

    hits=hits_trackML.rename(columns={'layer_id': 'layer', 'particle_id': 'track'})
    hits.track = hits.track.where(hits.track!=0, other=-1)
    hits = hits[np.logical_and(hits.volume_id==7, hits.module_id==1)]
    hits.reset_index(drop=True, inplace=True)
    hits['layer'] = hits.layer//2
    return hits 


def inputBMaN():
    import pandas as pd
    simdata = pd.read_csv('simdata_ArPb_3.2AGeV_mb_1.zip', sep='\t',
                          names=['event_id', 'x', 'y', 'z', 'detector_id', 'station_id', 'track_id', 'px', 'py', 'pz', 'vx', 'vy', 'vz'])
    return(simdata)


def transformBMaN(simdata):
    cc = simdata.groupby('event_id').x.count()
    cc[cc==cc[cc<=800].max()]
    simdata['layer'] = simdata.detector_id * 3 + simdata.station_id
    return simdata.rename(columns={'track_id': 'track'})[['x', 'y', 'z', 'layer', 'track', 'event_id']]
