import pandas as pd


def get_datasets():
    return ['simple', 'bman', 'spdsim', 'trackml', 'trackml_volume']


def get_hits(dataset: str = 'simple', n_events=None, *args, **kwargs) -> pd.DataFrame:
    from datasets.bman import get_hits_bman, get_hits_bman_one_event
    from datasets.spdsim import get_hits_spdsim, get_hits_spdsim_one_event
    from datasets.simple import get_hits_simple, get_hits_simple_one_event
    from datasets.trackml import get_hits_trackml, get_hits_trackml_by_volume, \
        get_hits_trackml_one_event, get_hits_trackml_one_event_by_volume
    dataset = dataset.lower()
    if n_events == 1:
        getter = {
            'simple': get_hits_simple_one_event,
            'bman': get_hits_bman_one_event,
            'spdsim': get_hits_spdsim_one_event,
            'trackml': get_hits_trackml_one_event,
            'trackml_volume': get_hits_trackml_one_event_by_volume
        }[dataset]
        return getter(*args, **kwargs)
    else:
        getter = {
            'simple': get_hits_simple,
            'bman': get_hits_bman,
            'spdsim': get_hits_spdsim,
            'trackml': get_hits_trackml,
            'trackml_volume': get_hits_trackml_by_volume
        }[dataset]
        return getter(n_events, *args, **kwargs)
