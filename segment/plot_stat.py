from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from pathos.multiprocessing import ProcessPool

from segment.stat import stat_seg_neighbors


def plot_stat_seg_neighbors(hits: pd.DataFrame) -> Artist:
    max_r = np.sqrt((hits.x.max() - hits.x.min()) ** 2 +
                    (hits.y.max() - hits.y.min()) ** 2 +
                    (hits.z.max() - hits.z.min()) ** 2)
    df = stat_seg_neighbors(hits, 0, max_r, 100, ProcessPool, cpu_count())
    df = df.melt(id_vars=['r', 'event'],
                 value_vars=['seg_all', 'seg_diff_level'],
                 var_name='filter', value_name='number_of_segments')
    plot = sns.lineplot(data=df, x='r', y='number_of_segments', hue='filter', style='filter')
    every_pair = (hits.groupby('event_id').size() ** 2).mean()
    plt.hlines(every_pair, df.r.min(), df.r.max(), color='black', linestyles='dashdot', linewidths=1.)
    return plot


if __name__ == '__main__':
    from datasets import get_hits

    event = get_hits('simple', 1)
    plot_stat_seg_neighbors(event)
    plt.show()
