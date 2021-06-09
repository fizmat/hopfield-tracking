import math
from typing import Tuple, List, Generator

import numpy as np
from numpy.random import default_rng


class SimpleEventGenerator:
    def __init__(self, seed=None):
        self.rng = default_rng(seed)

    def gen_one_track(self) -> Tuple[float, float, float]:
        x = self.rng.uniform(math.cos(math.radians(15)), 1., 1)
        alpha = self.rng.uniform(0.0, 2 * math.pi, 1)
        r = math.sqrt(1. - x ** 2)
        y = r * math.cos(alpha)
        z = r * math.sin(alpha)
        return x, y, z

    def gen_event_tracks(self, n: int = 10) -> np.ndarray:
        xyz = np.empty((n, 3))

        for i in range(n):
            while True:
                x, y, z = self.gen_one_track()
                xyz[i] = x, y, z
                if np.all(np.dot(xyz[:i], xyz[i]) < math.cos(0.05)):
                    break
        return xyz

    def gen_event_hits(self, xyz: np.ndarray) -> List[np.ndarray]:
        layers = []
        for i in range(8):
            x_detector = 0.5 + 0.5 * i
            xs = xyz[:, 0][:, None]
            hits = xyz / xs * x_detector
            hits[:, 1] += self.rng.normal(0, 0.005, hits[:, 1].shape)
            hits[:, 2] += self.rng.normal(0, 0.005, hits[:, 2].shape)
            layers.append(hits)
        return layers

    def gen_many_events(self, n: int = 1000, event_size: int = 10) -> Generator[List[np.ndarray], None, None]:
        for _ in range(n):
            yield self.gen_event_hits(self.gen_event_tracks(event_size))
