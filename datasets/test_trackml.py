import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from datasets.trackml import _read_truth, _read_blacklist, _transform, get_hits_trackml, get_hits_trackml_by_module, \
    get_hits_trackml_by_volume

_test_event = pd.DataFrame({
    'hit_id': [1, 2, 3, 4, 5],
    'x': [1., 2., 3., 4., 5.],
    'y': [2., 3., 4., 5., 6.],
    'z': [3., 4., 5., 6., 7.],
    'volume_id': [7] * 5,
    'layer_id': [2] * 5,
    'module_id': [1] * 5,
    'particle_id': [0, 123456789012345678, 0, 123456789012345678, 987654321098765432],
    'tx': [0.1, 0.2, 0.3, 0.4, 0.5],
    'ty': [0.2, 0.3, 0.4, 0.5, 0.6],
    'tz': [0.3, 0.4, 0.5, 0.6, 0.7],
    'tpx': [11, 1.1, 22, 1.2, 1.3],
    'tpy': [22, 2.2, 33, 2.3, 2.4],
    'tpz': [33, 3.3, 44, 3.4, 3.5],
    'weight': [0, 1e-6, 0, 2e-6, 3e-6]
})

_test_blacklist = pd.DataFrame({'hit_id': [1, 5]})


def test__read_truth(tmp_path):
    d = tmp_path / 'test_datasets_trackml'
    d.mkdir()
    hits = ('hit_id,x,y,z,volume_id,layer_id,module_id\n' +
            '1,1.0,2.0,3.0,7,2,1\n' +
            '2,2.0,3.0,4.0,7,2,1\n' +
            '3,3.0,4.0,5.0,7,2,1\n' +
            '4,4.0,5.0,6.0,7,2,1\n' +
            '5,5.0,6.0,7.0,7,2,1\n')
    truth = ('hit_id,particle_id,tx,ty,tz,tpx,tpy,tpz,weight\n' +
             '1,0,0.1,0.2,0.3,11,22,33,     0\n' +
             '2,123456789012345678,0.2,0.3,0.4,1.1,2.2,3.3,1e-06\n' +
             '3,0,0.3,0.4,0.5,22,33,44,     0\n' +
             '4,123456789012345678,0.4,0.5,0.6,1.2,2.3,3.4,2e-06\n' +
             '5,987654321098765432,0.5,0.6,0.7,1.3,2.4,3.5,3e-06\n')
    (d / 'test-hits.csv').write_text(hits)
    (d / 'test-truth.csv').write_text(truth)
    events = _read_truth('test', str(d))
    assert_frame_equal(events, _test_event)


def test__read_blacklist(tmp_path):
    d = tmp_path / 'test_datasets_trackml'
    d.mkdir()
    (d / 'test-blacklist_hits.csv').write_text('hit_id\n1\n5\n')
    assert_frame_equal(_read_blacklist('test', str(d)), _test_blacklist)


def test__transform():
    events = _transform(_test_event, _test_blacklist)
    assert_frame_equal(events, pd.DataFrame({
        'hit_id': [2, 3, 4],
        'x': [2., 3., 4.],
        'y': [3., 4., 5.],
        'z': [4., 5., 6.],
        'volume_id': [7] * 3,
        'layer': [1] * 3,
        'module_id': [1] * 3,
        'track': [123456789012345678, -1, 123456789012345678],
        'tx': [0.2, 0.3, 0.4],
        'ty': [0.3, 0.4, 0.5],
        'tz': [0.4, 0.5, 0.6],
        'tpx': [1.1, 22, 1.2],
        'tpy': [2.2, 33, 2.3],
        'tpz': [3.3, 44, 3.4],
        'weight': [1e-6, 0, 2e-6]
    }))


@pytest.mark.slow
@pytest.mark.bman
def test_get_hits_trackml():
    events = get_hits_trackml()
    assert_array_equal(events.index, range(10952747))
    assert_array_equal(events.event_id.unique(), range(1000, 1100))
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


@pytest.mark.slow
@pytest.mark.bman
def test_get_hits_trackml_by_volume():
    events = get_hits_trackml_by_volume()
    assert_array_equal(events.index, range(10952747))
    assert_array_equal(events.event_id.str.fullmatch(r'\d{4}-\d{1,2}'), True)
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


@pytest.mark.slow
@pytest.mark.bman
def test_get_hits_trackml_by_module():
    events = get_hits_trackml_by_module()
    assert_array_equal(events.index, range(10952747))
    assert_array_equal(events.event_id.str.fullmatch(r'\d{4}-\d{1,2}-\d{1,4}'), True)
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'
