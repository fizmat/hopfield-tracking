import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from datasets.bman import _read, get_hits_bman, _transform, get_hits_bman_one_event

_test_event = pd.DataFrame(data=[[4, 0.5, 0.6, 0.7, 5, 6, 7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4],
                                 [4, 0.4, 0.5, 0.6, 5, 5, 7, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3]],
                           columns=['event_id', 'x', 'y', 'z',
                                    'detector_id', 'station_id', 'track_id',
                                    'px', 'py', 'pz', 'vx', 'vy', 'vz'])


def test__read(tmp_path):
    d = tmp_path / 'test_datasets_bman'
    d.mkdir()
    s = ('4 0.5 0.6 0.7 5 6 7 0.8 0.9 1.1 1.2 1.3 1.4\n' +
         '4 0.4 0.5 0.6 5 5 7 0.7 0.8 0.9 1.1 1.2 1.3').replace(' ', '\t')
    (d / 'test.txt').write_text(s)
    events = _read(str(d), 'test.txt')
    assert_frame_equal(events, _test_event)


def test__transform():
    assert_frame_equal(_transform(_test_event),
                       pd.DataFrame(data=[[4, 0.5, 0.6, 0.7, 15 + 6, 7],
                                          [4, 0.4, 0.5, 0.6, 15 + 5, 7]],
                                    columns=['event_id', 'x', 'y', 'z', 'layer', 'track'])
                       )


@pytest.mark.slow
@pytest.mark.bman
def test_get_hits_bman():
    events = get_hits_bman()
    assert_array_equal(events.index, range(15813216))
    assert_array_equal(events.event_id.unique(), range(25000))
    assert set(events.layer.unique()) == set(range(9))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


def test_get_hits_bman_one_event():
    event6 = get_hits_bman_one_event()
    assert_array_equal(event6.index, range(858))
    assert_array_equal(event6.event_id, [6] * 858)
    assert set(event6.layer.unique()) == set(range(9))
    assert event6.track.min() == -1
    assert event6.track.dtype == 'int64'
