from typing import Tuple

import numpy as np
import pandas as pd


def cylindric_coordinates(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    return r, phi


def add_cylindric_coordinates(hits: pd.DataFrame) -> pd.DataFrame:
    hits['r'], hits['phi'] = cylindric_coordinates(hits.x, hits.y)
    return hits
