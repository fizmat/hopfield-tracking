import math
import random

import numpy as np


def gen_one_track():
    x = random.uniform(math.cos(math.radians(15)), 1.)
    alpha = random.uniform(0.0, 2 * math.pi)
    r = math.sqrt(1. - x ** 2)
    y = r * math.cos(alpha)
    z = r * math.sin(alpha)
    return x, y, z


def gen_event_tracks(n=10):
    xyz = np.empty((n, 3))

    for i in range(n):
        while True:
            x, y, z = gen_one_track()
            xyz[i] = x, y, z
            if np.all(np.dot(xyz[:i], xyz[i]) < math.cos(0.05)):
                break
    return xyz


def gen_event_hits(xyz):
    layers = []
    for i in range(8):
        x_detector = 0.5 + 0.5 * i
        xs = xyz[:, 0][:, None]
        hits = xyz / xs * x_detector
        hits[:, 1] += np.random.normal(0, 0.005, hits[:, 1].shape)
        hits[:, 2] += np.random.normal(0, 0.005, hits[:, 2].shape)
        layers.append(hits)
    return layers


def gen_many_events(n=1000, event_size=10):
  for _ in range(n):
    yield gen_event_hits(gen_event_tracks(event_size))

