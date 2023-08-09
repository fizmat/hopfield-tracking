import math
from typing import Tuple, Generator, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng

LAYER_DIST = 0.5


def get_hits(n_events: Optional[int] = 100, noisiness=1., event_size: Optional[int] = 10,
             seed=1) -> pd.DataFrame:
    if n_events is None:
        n_events = 100
    if event_size is None:
        event_size = 10
    hits_list = []
    for i, event in enumerate(SimpleEventGenerator(
            seed=seed, noisiness=noisiness).gen_many_events(n_events, event_size)):
        event['event_id'] = i
        hits_list.append(event)
    return pd.concat(hits_list, ignore_index=True)


def get_one_event(event_size=10, noisiness=1., seed=1):
    hits = next(SimpleEventGenerator(seed=seed, noisiness=noisiness).gen_many_events(1, event_size))
    hits['event_id'] = 0
    return hits


class SimpleEventGenerator:
    def __init__(self, halfwidth_degrees: float = 15, n_layers: int = 8,
                 field_strength=1., xy_hit_deviation=0.005, noisiness=1., seed=1, box_size=None):
        self.halfwidth = math.radians(halfwidth_degrees)
        self.rng = default_rng(seed)
        self.n_layers = n_layers
        self.layers_thickness = LAYER_DIST
        self.field_strength = field_strength
        self.xy_hit_deviation = xy_hit_deviation
        self.noisiness = noisiness
        self.layer_i = np.arange(n_layers)
        self.layer_z = self.layers_thickness + self.layers_thickness * self.layer_i
        if box_size is None:
            self.size_x = self.size_y = self.layer_z[-1] * np.sin(self.halfwidth)
        else:
            self.size_x = self.size_y = box_size

    def gen_directions_in_cone(self, n: int = 1) -> ndarray:
        z = self.rng.uniform(math.cos(self.halfwidth), 1., n)
        alpha = self.rng.uniform(0.0, 2 * math.pi, n)
        r = np.sqrt(1. - z ** 2)
        x = r * np.cos(alpha)
        y = r * np.sin(alpha)
        return np.stack([x, y, z], axis=1)

    def run_straight_track(self, m: np.ndarray) -> ndarray:
        track_vector = m / m[2]  # scale track vector to put z-component at 1
        hits = track_vector[None, :] * self.layer_z[:, None]  # [layer, coordinate]
        return hits

    def find_track_spiral_circle(self, m: np.ndarray, q: float) -> Tuple[ndarray, float]:
        m_xy = m[:2]
        r = np.linalg.norm(m_xy) / abs(self.field_strength * q)
        # to_center = np.array([[0, 1], [0, -1]]).dot(m_xy) / m ?
        center = np.array([[0, -1], [1, 0]]).dot(m_xy) / (self.field_strength * q)  # r * to_center
        return center, r

    def run_curved_track(self, m: np.ndarray, q: float) -> ndarray:
        m_z = m[2]
        center, r = self.find_track_spiral_circle(m, q)
        # TC = 2*pi*r/v_xy  TL = h/v_z
        arc_per_dist = self.layers_thickness * self.field_strength * q / m_z
        angles = np.arctan2(-center[1], -center[0]) + arc_per_dist * self.layer_z
        hits = np.stack([center[0] + r * np.cos(angles), center[1] + r * np.sin(angles), self.layer_z], axis=-1)
        return hits

    def run_track(self, momentum: np.ndarray, charge: float) -> Tuple[ndarray, ndarray]:
        if self.field_strength != 0 and charge != 0:
            hits = self.run_curved_track(momentum, charge)
        else:
            hits = self.run_straight_track(momentum)
        hits[:, :2] += self.rng.normal(0, self.xy_hit_deviation, hits[:, :2].shape)  # inaccuracy in x and y
        return hits, self.layer_i

    def gen_tracks(self, momenta: np.ndarray, charges: np.ndarray
                   ) -> Generator[Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray], None, None]:
        for i, (m, q) in enumerate(zip(momenta, charges)):
            h, layers = self.run_track(m, q)
            k = len(layers)
            yield *h.T, layers, np.full(k, i), np.full(k, q)

    def gen_noise(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        noise_count = self.rng.poisson(self.noisiness)
        noise_layer = self.rng.integers(0, self.n_layers, noise_count)
        return (self.rng.uniform(-self.size_x, self.size_x, noise_count),
                self.rng.uniform(-self.size_y, self.size_y, noise_count),
                self.layer_z[noise_layer],
                noise_layer,
                np.full(noise_count, -1),
                np.full(noise_count, np.nan))

    def gen_event(self, momenta: np.ndarray, charges: np.ndarray) -> pd.DataFrame:
        columns = map(np.concatenate, zip(*self.gen_tracks(momenta, charges),
                                          *[self.gen_noise()]))
        return pd.DataFrame(dict(zip(['x', 'y', 'z', 'layer', 'track', 'charge'], columns)))

    def gen_many_events(self, n: int = 1000, event_size: int = 10, momentum_scale=.2) \
            -> Generator[pd.DataFrame, None, None]:
        for _ in range(n):
            yield self.gen_event(
                self.rng.gamma(shape=5., scale=momentum_scale, size=event_size)[:, np.newaxis]
                * self.gen_directions_in_cone(event_size),
                charges=self.rng.choice([1, -1], event_size)
            )


def main():
    get_hits(10000)


if __name__ == "__main__":
    main()
