#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import random
import math
from typing import Tuple

import numpy as np
from numpy import pi
import pandas as pd


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


def get_hits_spdsim_one_event(max_ntrk=10):
    return get_hits_spdsim(1, max_ntrk)


def get_hits_spdsim(n_events=None, max_ntrk=10):
    if n_events is None:
        n_events = 100
    return gen_spdsim(n_events, max_ntrk).rename(
        columns={'station': 'layer', 'evt': 'event_id', 'trk': 'track'}
    )[['x', 'y', 'z', 'layer', 'track', 'event_id']]


def gen_spdsim(n_events=100, max_ntrk=10):
    # track_coords_all = []
    eff = 1  # detector efficiency

    radii = np.linspace(270, 850, 35)  # mm

    records = []
    for evt in range(0, n_events):
        vtxx = random.gauss(0, 10)
        vtxy = random.gauss(0, 10)
        vtxz = random.uniform(-300, 300)  # mm
        ntrk = int(random.uniform(1, max_ntrk))
        for trk in range(0, ntrk):

            pt = random.uniform(100, 1000)  # MeV/c
            phi = random.uniform(0, 2 * pi)
            theta = math.acos(random.uniform(-1, 1))

            charge = random.choice((-1, 1))

            stations, x, y, z, px, py, pz = extrapolate_to_r(pt, charge, theta, phi, vtxz, radii)
            for i, station in enumerate(stations):
                if z[i] >= 2386 or z[i] <= -2386:
                    continue
                if random.uniform(0, 1) > eff:
                    continue
                records.append((evt, x[i], y[i], z[i], station, trk, px[i], py[i], pz, vtxx, vtxy, vtxz))

        # add noise hits
        nhit = int(random.uniform(10, 100))  # up to 100 noise hits
        sta = np.random.randint(0, 35, nhit)
        r = radii[sta]
        phi = np.random.uniform(0, 2 * pi, nhit)
        z = np.random.uniform(-2386, 2386, nhit)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        for i in range(nhit):
            records.append((evt, x[i], y[i], z[i], sta[i], -1, 0, 0, 0, 0, 0, 0))

    hits = pd.DataFrame(records,
                        columns=['evt', 'x', 'y', 'z', 'station', 'trk',
                                 'px', 'py', 'pz', 'vtxx', 'vtxy', 'vtxz'])
    hits.z += np.random.normal(0, 0.1, len(hits))
    phit = np.arctan2(hits.x, hits.y)
    delta = np.random.normal(0, 0.1, len(hits))
    hits.x += delta * np.sin(phit)
    hits.y -= delta * np.cos(phit)
    return hits


if __name__ == '__main__':
    hits = gen_spdsim(int(sys.argv[1]))
    hits.to_csv('output.tsv', sep='\t', index=False, header=False)
