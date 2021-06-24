import math
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng


class SimpleEventGenerator:
    def __init__(self, halfwidth_degrees: float = 15, n_layers: int = 8, layers_thickness=0.5,
                 field_strength=1., seed=None):
        self.halfwidth = math.radians(halfwidth_degrees)
        self.rng = default_rng(seed)
        self.n_layers = n_layers
        self.layers_thickness = layers_thickness
        self.field_strength = field_strength
        self.layer_i = np.arange(n_layers)
        self.layer_x = self.layers_thickness + self.layers_thickness * self.layer_i

    def gen_directions_in_cone(self, n: int = 1) -> ndarray:
        x = self.rng.uniform(math.cos(self.halfwidth), 1., n)
        alpha = self.rng.uniform(0.0, 2 * math.pi, n)
        r = np.sqrt(1. - x ** 2)
        y = r * np.cos(alpha)
        z = r * np.sin(alpha)
        return np.stack([x, y, z], axis=1)

    def gen_momentum(self, n: int = 1) -> ndarray:
        return self.rng.gamma(shape=5., scale=1., size=n)

    def run_straight_track(self, m: np.ndarray) -> ndarray:
        track_vector = m / m[0]  # scale track vector to put x-component at 1
        hits = track_vector[None, :] * self.layer_x[:, None]  # [layer, coordinate]
        return hits

    def run_curved_track(self, m: np.ndarray):
        m_yz = np.linalg.norm(m[1:])
        r = m_yz / self.field_strength
        # to_center = np.array([[0, 1], [0, -1]]).dot(momentum[1:]) / m
        center = np.array([[0, -1], [1, 0]]).dot(m[1:]) / self.field_strength  # r * to_center
        # TC = 2*pi*r/v_xy  TL = h/v_z
        arc_per_dist = self.layers_thickness * self.field_strength / m[0]
        angles = np.arctan2(m[2], m[1]) + arc_per_dist * self.layer_x
        hits = np.stack([self.layer_x, center[0] + r * np.sin(angles), center[1] - r * np.cos(angles)], axis=-1)
        return hits

    def run_track(self, momentum: np.ndarray) -> Tuple[pd.DataFrame, ndarray]:
        if self.field_strength > 0:
            hits = self.run_curved_track(momentum)
        else:
            hits = self.run_straight_track(momentum)
        hits[:, 1:] += self.rng.normal(0, 0.005, hits[:, 1:].shape)  # inaccuracy in y and z
        data = np.concatenate([hits, self.layer_i[:, np.newaxis]], axis=1)
        ii = self.layer_i[:-1]
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
