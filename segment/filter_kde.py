import holoviews as hv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity

from datasets import get_hits
from segment.track import gen_seg_track_sequential
from tracking.hit import add_cylindric_coordinates


def seg_angle(cyl, seg):
    starts = cyl[seg[:, 0]]
    ends = cyl[seg[:, 1]]
    r1 = starts[:, 0]
    r2 = ends[:, 0]
    z1 = starts[:, 1]
    z2 = ends[:, 1]
    angle = np.arctan2((z2 - z1), (r2 - r1))
    with np.errstate(divide='ignore', invalid='ignore'):
        intercept = z1 - (z2 - z1) * r1 / (r2 - r1)
    return angle, intercept


def get_gaussian(events):
    all_g = []
    for ei, ev in events.groupby(by='event_id'):
        hits = ev.reset_index(drop=True)
        seg = gen_seg_track_sequential(hits)
        cyl = hits[['r', 'z', 'phi']].values
        angle, intercept = seg_angle(cyl, seg)
        gaussian = pd.DataFrame({'angle': angle, 'intercept': intercept})
        all_g.append(gaussian)
    gaussian = pd.concat(all_g, ignore_index=True)
    gaussian['directed_angle'] = gaussian.angle
    gaussian.angle = gaussian.angle.where(gaussian.angle > -np.pi / 2, np.pi + gaussian.angle)
    gaussian.angle = gaussian.angle.where(gaussian.angle < np.pi / 2, gaussian.angle - np.pi)
    return gaussian


def _profile():
    min_angle = -2
    max_angle = 2
    min_intercept = -20
    max_intercept = 20
    size_portion = 1.0
    nx = 201
    ny = 201

    bandwidth = 0.2
    rtol = 0.1
    atol = 0

    hits = get_hits('trackml', 2)
    add_cylindric_coordinates(hits)
    gaussian = get_gaussian(hits)
    gaussian[np.abs(gaussian.intercept) < 1e4].plot.scatter('angle', 'intercept')
    plt.show()
    with pd.option_context('mode.use_inf_as_na', True):
        gaussian.dropna(inplace=True)

    gaussian.intercept.hist(bins=np.logspace(np.log10(1e-10), np.log10(1e14), 50), log=True).set_xscale("log")
    plt.show()
    gaussian.intercept.hist(bins=np.logspace(np.log10(1e-10), np.log10(1e14), 200),
                            log=True, density=True, cumulative=-1,
                            figsize=(10, 8)
                            ).set_xscale("log")
    plt.xticks(np.logspace(np.log10(1e-10), np.log10(1e14), 25))
    plt.show()
    gaussian[np.abs(gaussian.intercept) < max_intercept].plot.hexbin('directed_angle', 'intercept',
                                                           bins='log',
                                                           gridsize=256, figsize=(16, 12), sharex=False)
    plt.show()
    sns.jointplot(
        x="angle",
        y="intercept",
        kind="hist",
        bins='doane',
        data=gaussian[np.abs(gaussian.intercept) < max_intercept]
    )
    gaussian[np.abs(gaussian.intercept) < max_intercept].plot.hexbin('angle', 'intercept', gridsize=256, bins='log',
                                                           figsize=(16, 12), sharex=False)
    plt.show()

    portion = gaussian[np.abs(gaussian.intercept) < max_intercept * 10].copy()
    portion.intercept /= 10
    X = portion.sample(frac=size_portion)[['angle', 'intercept']].to_numpy()
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, rtol=rtol, atol=atol).fit(X)
    xx = np.mgrid[min_angle:max_angle:nx * 1j, -max_intercept:max_intercept:ny * 1j].reshape((2, -1)).T
    zz = kde.score_samples(xx)
    plt.imshow(zz.reshape(nx, ny).T, extent=[min_angle, max_angle, -max_intercept, max_intercept],
               aspect='auto')
    plt.colorbar()
    plt.show()
    plt.imshow(zz.reshape(nx, ny).T > -6, extent=[min_angle, max_angle, -max_intercept, max_intercept],
               aspect='auto')
    plt.show()

    hv.extension('matplotlib')
    hv.Bivariate(X).opts(colorbar=True, cmap='Blues', filled=True)
    plt.show()
    zz1 = pd.DataFrame(data=zz.reshape(nx, ny))
    zz1.to_csv(r"zz.csv", index=False, sep=',')
    size_pixel_x = (max_angle - min_angle) / (nx - 1)
    size_pixel_y = (max_intercept - min_intercept) / (ny - 1)

    gaussian['pixel_x'] = ((gaussian.angle - min_angle) // size_pixel_x).astype(int)
    gaussian['pixel_y'] = (((gaussian.intercept / 10) - min_intercept) // size_pixel_y).astype(int)

    picture = zz.reshape(nx, ny).T
    condition = np.logical_and(gaussian.pixel_y >= 0, gaussian.pixel_y <= (ny - 1))
    good = gaussian[condition]

    gaussian.loc[condition, 'kde'] = picture[good.pixel_y, good.pixel_x]
    gaussian.loc[gaussian.pixel_y < 0, 'kde'] = gaussian[condition].kde.min()
    gaussian.loc[gaussian.pixel_y > (ny - 1), 'kde'] = gaussian[condition].kde.min()
    gaussian.kde.hist(bins=50, density=True, cumulative=True)
    gaussian.kde.hist(bins=200,
                      log=True, density=True, cumulative=1,
                      figsize=(10, 8)
                      )
    plt.yticks(np.logspace(np.log10(1e-2), 0, 9), np.logspace(np.log10(1e-2), 0, 9))
    plt.show()


if __name__ == '__main__':
    _profile()
