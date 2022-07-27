import holoviews as hv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity

from datasets import get_hits
from segment.track import gen_seg_track_sequential
from tracking.hit import add_cylindric_coordinates


def add_angle_intecept(hits: pd.DataFrame, seg: np.ndarray, axis='z', distance='r') -> pd.DataFrame:
    r1, r2 = (hits.loc[seg[:, i], distance].to_numpy() for i in (0, 1))
    z1, z2 = (hits.loc[seg[:, i], axis].to_numpy() for i in (0, 1))
    directed_angle = np.arctan2((z2 - z1), (r2 - r1))
    with np.errstate(divide='ignore', invalid='ignore'):
        intercept = z1 - (z2 - z1) * r1 / (r2 - r1)
    angle = np.where(directed_angle <= -np.pi / 2, np.pi + directed_angle, directed_angle)
    angle = np.where(angle > np.pi / 2, angle - np.pi, angle)
    segments = pd.DataFrame(seg, columns=['start', 'end'])
    segments['angle'] = angle
    segments['intercept'] = intercept
    segments['directed_angle'] = directed_angle
    return segments


def plot_angle_intercept(gaussian: pd.DataFrame, max_intercept: float) -> None:
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


def plot_angle_intercept_array(X: np.ndarray) -> None:
    mr = hv.renderer('matplotlib')
    plot = hv.Bivariate(X).opts(colorbar=True, cmap='Blues', filled=True)
    mr.show(plot)


def plot_kde_grid(zz: np.ndarray, nx: int, ny: int,
                  min_angle: float, max_angle: float, min_intercept: float, max_intercept: float,
                  limit: float) -> None:
    plt.imshow(zz.reshape(nx, ny).T,
               extent=[min_angle, max_angle, min_intercept, max_intercept], aspect='auto')
    plt.colorbar()
    plt.show()
    plt.imshow(zz.reshape(nx, ny).T > limit,
               extent=[min_angle, max_angle, min_intercept, max_intercept], aspect='auto')
    plt.show()


def apply_kde_grid(gaussian, max_angle, max_intercept, min_angle, min_intercept, nx, ny, zz) -> pd.DataFrame:
    size_pixel_x = 2 * max_angle / (nx - 1)
    size_pixel_y = (max_intercept - min_intercept) / (ny - 1)
    gaussian['pixel_x'] = ((gaussian.angle - min_angle) // size_pixel_x).astype(int)
    gaussian['pixel_y'] = (((gaussian.intercept / 10) - min_intercept) // size_pixel_y).astype(int)
    picture = zz.reshape(nx, ny).T
    condition = np.logical_and(gaussian.pixel_y >= 0, gaussian.pixel_y <= (ny - 1))
    good = gaussian[condition]
    gaussian.loc[condition, 'kde'] = picture[good.pixel_y, good.pixel_x]
    gaussian.loc[gaussian.pixel_y < 0, 'kde'] = gaussian[condition].kde.min()
    gaussian.loc[gaussian.pixel_y > (ny - 1), 'kde'] = gaussian[condition].kde.min()
    return gaussian


def plot(hits: pd.DataFrame, max_intercept=20, min_intercept=None, nx=201, ny=201,
         bandwidth=0.2, rtol=0.1, atol=0, max_angle=2, min_angle=None) -> None:
    if min_intercept is None:
        min_intercept = -max_intercept
    if min_angle is None:
        min_angle = -max_angle

    seg = np.concatenate([s for s in hits.groupby('event_id').apply(gen_seg_track_sequential) if len(s)])
    add_cylindric_coordinates(hits)
    segments = add_angle_intecept(hits, seg)
    plot_angle_intercept(segments, max_intercept)
    portion = segments[np.abs(segments.intercept) < max_intercept * 10].copy()
    portion.intercept /= 10
    X = portion[['angle', 'intercept']].to_numpy()
    plot_angle_intercept_array(X)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, rtol=rtol, atol=atol)
    kde.fit(X)
    xx = np.mgrid[-max_angle:max_angle:nx * 1j, -max_intercept:max_intercept:ny * 1j].reshape((2, -1)).T
    zz = kde.score_samples(xx)
    plot_kde_grid(zz, nx, ny, min_angle, max_angle, min_intercept, max_intercept, -6)
    pd.DataFrame(data=zz.reshape(nx, ny)).to_csv(r"zz.csv", index=False, sep=',')
    apply_kde_grid(segments, max_angle, max_intercept, min_angle, min_intercept, nx, ny, zz)
    segments.kde.hist(bins=50, density=True, cumulative=True)
    plt.show()
    segments.kde.hist(bins=200,
                      log=True, density=True, cumulative=1,
                      figsize=(10, 8)
                      )
    plt.yticks(np.logspace(np.log10(1e-2), 0, 9), np.logspace(np.log10(1e-2), 0, 9))
    plt.show()


if __name__ == '__main__':
    hits = get_hits('trackml', 10)
    plot(hits)
