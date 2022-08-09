import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from metrics.segments import gen_perfect_act

_hits = pd.DataFrame({'track': [1, -1, 3, 5, -1, 3, 5, 1, 3],
                      'layer': [0, 0, 0, 1, 1, 1, 2, 2, 2]})

_seg = np.array([(0, 3), (0, 4), (0, 5),
                 (1, 3), (1, 4), (1, 5),
                 (2, 3), (2, 4), (2, 5),
                 (3, 6), (3, 7), (3, 8),
                 (4, 6), (4, 7), (4, 8),
                 (5, 6), (5, 7), (5, 8)])

_tseg = np.array([(2, 5), (5, 8), (3, 6)])


def test_gen_perfect_act():
    assert_array_equal(gen_perfect_act(_seg, _tseg), [0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                      1, 0, 0, 0, 0, 0, 0, 0, 1])
