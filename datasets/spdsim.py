#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from numpy import pi

LAYER_DIST = 35.  # mm


def extrapolate_to_r(pt: float, charge: float, theta: float, phi: float, z0: float, rc: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    b = 0.8  # magnetic field [T}
    stations = np.arange(1, len(rc) + 1)

    pz = pt / math.tan(theta) * charge

    phit = phi - pi / 2
    r = pt / 0.29 / b  # mm
    k0 = r / math.tan(theta)
    x0 = r * math.cos(phit)
    y0 = r * math.sin(phit)

    is_intersection = 2 * r >= rc
    rc = rc[is_intersection]
    stations = stations[is_intersection]

    r *= charge  # both polarities
    alpha = 2 * np.arcsin(rc / (2 * r))

    not_spinning_track = alpha <= pi
    rc = rc[not_spinning_track]  # algorithm doesn't work for spinning tracks
    stations = stations[not_spinning_track]
    alpha = alpha[not_spinning_track]

    extphi = (phi - alpha / 2) % 2 * pi

    x = rc * np.cos(extphi)
    y = rc * np.sin(extphi)

    rax, ray = x - x0 * charge, y - y0 * charge

    tax, tay = -ray, rax

    tabs = np.sqrt(tax * tax + tay * tay)  # pt
    tax /= tabs
    tay /= tabs
    tax *= -pt * charge
    tay *= -pt * charge

    z = z0 + k0 * alpha
    return stations, x, y, z, tax, tay, pz


def get_hits_spdsim_one_event(event_size=10, efficiency=1., n_noise_hits=100, seed=1):
    return get_hits_spdsim(1, event_size, efficiency, n_noise_hits, seed)


def get_hits_spdsim(n_events: Optional[int] = 100, event_size: Optional[int] = 10,
                    efficiency: float = 1., n_noise_hits: int = 100, seed=1) -> pd.DataFrame:
    if n_events is None:
        n_events = 100
    if event_size is None:
        event_size = 10
    return gen_spdsim(n_events, event_size, efficiency, n_noise_hits, seed).rename(
        columns={'station': 'layer', 'evt': 'event_id', 'trk': 'track'}
    )[['x', 'y', 'z', 'layer', 'track', 'event_id']]


def gen_spdsim(n_events=100, event_size=10, efficiency=1., n_noise_hits=100, seed=1):
    radii = np.linspace(270, 850, 35)  # mm
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    np_rng2 = np.random.default_rng(seed)  # for coordinate measurement error, should not depend on n_events

    records = []
    for evt in range(0, n_events):
        vtxx = rng.gauss(0, 10)
        vtxy = rng.gauss(0, 10)
        vtxz = rng.uniform(-300, 300)  # mm
        ntrk = event_size
        for trk in range(0, ntrk):
            while True:
                pt = rng.uniform(100, 1000)  # MeV/c
                phi = rng.uniform(0, 2 * pi)
                theta = math.acos(rng.uniform(-1, 1))
                charge = rng.choice((-1, 1))
                stations, x, y, z, px, py, pz = extrapolate_to_r(pt, charge, theta, phi, vtxz, radii)
                if np.logical_and(z < 2386, z > -2386).sum() > 5:
                    break
            for i, station in enumerate(stations):
                if z[i] >= 2386 or z[i] <= -2386:
                    continue
                if rng.uniform(0, 1) > efficiency:
                    continue
                records.append((evt, x[i], y[i], z[i], station, trk, px[i], py[i], pz, vtxx, vtxy, vtxz))

        nhit = n_noise_hits
        sta = np_rng.integers(0, 35, nhit)
        r = radii[sta]
        phi = np_rng.uniform(0, 2 * pi, nhit)
        z = np_rng.uniform(-2386, 2386, nhit)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        for i in range(nhit):
            records.append((evt, x[i], y[i], z[i], sta[i], -1, 0, 0, 0, 0, 0, 0))

    hits = pd.DataFrame(records,
                        columns=['evt', 'x', 'y', 'z', 'station', 'trk',
                                 'px', 'py', 'pz', 'vtxx', 'vtxy', 'vtxz'])

    def add_measurement_error(hits):
        hits.z += np_rng2.normal(0, 0.1, len(hits))
        phit = np.arctan2(hits.x, hits.y)
        delta = np_rng2.normal(0, 0.1, len(hits))
        hits.x += delta * np.sin(phit)
        hits.y -= delta * np.cos(phit)
        return hits

    hits = hits.groupby('evt').apply(add_measurement_error)
    return hits


if __name__ == '__main__':
    hits = gen_spdsim(int(sys.argv[1]))
    hits.to_csv('output.tsv', sep='\t', index=False, header=False)
