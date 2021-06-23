import math
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng


class SimpleEventGenerator:
    def __init__(self, halfwidth_degrees: float = 15, n_layers: int = 8, layers_thickness=0.5, seed=None):
        self.halfwidth = math.radians(halfwidth_degrees)
        self.rng = default_rng(seed)
        self.n_layers = n_layers
        self.layers_thickness = layers_thickness

    def gen_directions_in_cone(self, n: int = 1) -> ndarray:
        x = self.rng.uniform(math.cos(self.halfwidth), 1., n)
        alpha = self.rng.uniform(0.0, 2 * math.pi, n)
        r = np.sqrt(1. - x ** 2)
        y = r * np.cos(alpha)
        z = r * np.sin(alpha)
        return np.stack([x, y, z], axis=1)

    def gen_momentum(self, n: int = 1) -> ndarray:
        return self.rng.gamma(shape=5., scale=1., size=n)

    def run_track(self, momentum: np.ndarray) -> Tuple[pd.DataFrame, ndarray]:
        layer_i = np.arange(self.n_layers)
        layer_x = self.layers_thickness + self.layers_thickness * layer_i
        track_vector = momentum / momentum[0]  # scale track vector to put x-component at 1
        hits = track_vector[None, :] * layer_x[:, None]  # [layer, coordinate]
        hits[:, 1:] += self.rng.normal(0, 0.005, hits[:, 1:].shape)  # inaccuracy in y and z
        data = np.concatenate([hits, layer_i[:, np.newaxis]], axis=1)
        ii = np.arange(self.n_layers - 1)
        seg = np.stack([ii, ii + 1], axis=1)
        return pd.DataFrame(data, columns=['x', 'y', 'z', 'layer']), seg

    def gen_event(self, momenta: np.ndarray) -> Tuple[pd.DataFrame, ndarray]:
        hits = []
        seg = []
        n = 0
        for i, m in enumerate(momenta):
            h, s = self.run_track(m)
            h.index += n
            s += n
            h['track'] = i
            hits.append(h)
            seg.append(s)
            n += len(h)
        return pd.concat(hits), np.concatenate(seg)

    def gen_many_events(self, n: int = 1000, event_size: int = 10) -> Generator[pd.DataFrame, None, None]:
        for _ in range(n):
            yield self.gen_event(self.gen_directions_in_cone(event_size))
