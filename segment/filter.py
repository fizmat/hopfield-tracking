import numpy as np
import pandas as pd


def seg_drop_same_layer(seg: np.ndarray, event: pd.DataFrame) -> np.ndarray:
    a = event.loc[seg[:, 0], 'layer'].to_numpy()
    b = event.loc[seg[:, 1], 'layer'].to_numpy()
    comp = a != b
    return seg[comp]
