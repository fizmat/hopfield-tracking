import numpy as np
import pandas as pd


def gen_seg_track_sequential(event: pd.DataFrame) -> np.ndarray:
    return np.concatenate([np.stack((g.index[:-1], g.index[1:]), axis=-1)
                           for track, g in event.groupby('track')
                           if track >= 0])


def gen_seg_track_layered(event: pd.DataFrame) -> np.ndarray:
    track_segments = []
    event = event.loc[event.track >= 0]
    d = {layer: g.groupby('track').groups
         for layer, g in event.groupby('layer')}
    for layer, g in d.items():
        for track, start_hits in g.items():
            for b in d.get(layer + 1, {}).get(track, ()):
                track_segments += [(a, b) for a in start_hits]
    return np.array(track_segments)


def main():
    from datasets.trackml import get_hits_trackml_one_event
    event = get_hits_trackml_one_event()
    gen_seg_track_layered(event)
    gen_seg_track_sequential(event)


if __name__ == '__main__':
    main()
