import math
from typing import Generator

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng


class SimpleEventGenerator:
    def __init__(self, halfwidth_degrees: float = 15, seed=None):
        self.halfwidth = math.radians(halfwidth_degrees)
        self.rng = default_rng(seed)

    def gen_directions_in_cone(self, n: int = 1) -> ndarray:
        x = self.rng.uniform(math.cos(self.halfwidth), 1., n)
        alpha = self.rng.uniform(0.0, 2 * math.pi, n)
        r = np.sqrt(1. - x ** 2)
        y = r * np.cos(alpha)
        z = r * np.sin(alpha)
        return np.stack([x, y, z], axis=1)

    def gen_momentum(self, n: int = 1) -> ndarray:
        return self.rng.gamma(shape=5., scale=1., size=n)

    def gen_event_hits(self, xyz: np.ndarray) -> pd.DataFrame:
        layer_i = np.arange(8)
        layer_x = 0.5 + 0.5 * layer_i
        xs = xyz[:, 0][:, None]
        xyz = xyz / xs  # scale track vector to put x-component at 1
        hits = xyz[None, :, :] * layer_x[:, None, None]
        hits[..., 1] += self.rng.normal(0, 0.005, hits[..., 1].shape)
        hits[..., 2] += self.rng.normal(0, 0.005, hits[..., 2].shape)
        return pd.concat([
            pd.concat([
                pd.DataFrame(layer, columns=['x', 'y', 'z']),
                pd.DataFrame({'layer': i, 'track': np.arange(len(layer))})
            ], axis=1)
            for i, layer in enumerate(hits)],
            ignore_index=True
        )

    def gen_many_events(self, n: int = 1000, event_size: int = 10) -> Generator[pd.DataFrame, None, None]:
        for _ in range(n):
            yield self.gen_event_hits(self.gen_directions_in_cone(event_size))
