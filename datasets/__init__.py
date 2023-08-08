import pandas as pd


def get_datasets():
    return ['simple', 'bman', 'spdsim', 'trackml', 'trackml_volume']


def get_hits(dataset: str = 'simple', n_events=None, *args, **kwargs) -> pd.DataFrame:
    from datasets import bman, simple, spdsim, trackml
    dataset = dataset.lower()
    if n_events == 1:
        getter = {
            'simple': simple.get_one_event,
            'bman': bman.get_one_event,
            'spdsim': spdsim.get_one_event,
            'trackml': trackml.get_one_event,
            'trackml_volume': trackml.get_one_event_by_volume
        }[dataset]
        return getter(*args, **kwargs)
    else:
        getter = {
            'simple': simple.get_hits,
            'bman': bman.get_hits,
            'spdsim': spdsim.get_hits,
            'trackml': trackml.get_sample,
            'trackml_volume': trackml.get_sample_by_volume
        }[dataset]
        return getter(n_events, *args, **kwargs)
