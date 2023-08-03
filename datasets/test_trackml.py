import dask.dataframe as dd
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from datasets.trackml import _transform, get_one_event_by_volume, get_sample_by_volume, get_one_event, \
    get_sample, gen_train_1, _csv_one_event, _feather_one_event, _zip_sample, _feather_sample, _blacklist_hits, \
    HITS_PARQUET

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


def test_transform():
    hits = _transform(_test_event)
    assert_frame_equal(hits, pd.DataFrame({
        'hit_id': [1, 2, 3, 4, 5],
        'x': [1., 2., 3., 4., 5.],
        'y': [2., 3., 4., 5., 6.],
        'z': [3., 4., 5., 6., 7.],
        'volume_id': [7] * 5,
        'layer': [1] * 5,
        'module_id': [1] * 5,
        'track': [-1, 123456789012345678, -1, 123456789012345678, 987654321098765432],
        'tx': [0.1, 0.2, 0.3, 0.4, 0.5],
        'ty': [0.2, 0.3, 0.4, 0.5, 0.6],
        'tz': [0.3, 0.4, 0.5, 0.6, 0.7],
        'tpx': [11, 1.1, 22, 1.2, 1.3],
        'tpy': [22, 2.2, 33, 2.3, 2.4],
        'tpz': [33, 3.3, 44, 3.4, 3.5],
        'weight': [0, 1e-6, 0, 2e-6, 3e-6],
    }))


def test_good_blacklist():
    result = _blacklist_hits(
        pd.DataFrame({'hit_id': [1, 2, 3],
                      'particle_id': [123, 234, 123]}),
        pd.DataFrame({'hit_id': [1, 3]}),
        pd.DataFrame({'particle_id': [123]}))
    assert_array_equal(result, pd.DataFrame({
        'hit_id': [1, 2, 3],
        'particle_id': [123, 234, 123],
        'blacklisted': [True, False, True]
    }))


def test_bad_blacklist():
    with pytest.raises(AssertionError):
        _blacklist_hits(
            pd.DataFrame({'hit_id': [1, 2, 3],
                          'particle_id': [123, 234, 55]}),
            pd.DataFrame({'hit_id': [1, 3]}),
            pd.DataFrame({'particle_id': [123]}))


def test_feather_event():
    fast = _feather_one_event()
    assert_frame_equal(fast, _csv_one_event())


@pytest.mark.slow
@pytest.mark.trackml
def test_feather_sample():
    fast = _feather_sample()  # fail fast when file is missing
    assert_frame_equal(fast, _zip_sample())


@pytest.mark.trackml
def test_get_hits_trackml():
    events = get_sample()
    assert_array_equal(events.index, range(10967467))
    assert_array_equal(events.event_id.unique(), range(1000, 1100))
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


@pytest.mark.slow
@pytest.mark.trackml_1
def test_get_hits_trackml_1():
    for i, event in enumerate(gen_train_1(n_events=200)):
        assert_array_equal(event.index, range(len(event)))
        assert_array_equal(event.event_id, i + 1000)
        assert set(event.layer.unique()) == set(range(1, 8))
        assert event.track.min() == -1
        assert event.track.dtype == 'int64'
    assert i == 199


def test_get_hits_trackml_one_event():
    events = get_one_event()
    assert_array_equal(events.index, range(120939))
    assert_array_equal(events.event_id.unique(), 1000)
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


@pytest.mark.trackml
def test_get_hits_trackml_by_volume():
    events = get_sample_by_volume()
    assert_array_equal(events.index, range(10967467))
    assert events.event_id.max() == 109918
    assert events.event_id.min() == 100007
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


@pytest.mark.trackml
def test_get_hits_by_volume_limited():
    events = get_sample_by_volume(n_events=200)
    assert_array_equal(events.index, range(2526083))
    assert events.event_id.max() == 102208
    assert events.event_id.min() == 100007
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


def test_get_hits_trackml_one_event_by_volume():
    events = get_one_event_by_volume()
    assert_array_equal(events.index, range(16873))
    assert_array_equal(events.event_id, 100007)
    assert set(events.layer.unique()) == set(range(1, 8))
    assert events.track.min() == -1
    assert events.track.dtype == 'int64'


@pytest.mark.trackml_1
def test_dask_head():
    df = dd.read_parquet(HITS_PARQUET).head(500)
    assert_frame_equal(_transform(df), get_one_event().iloc[:500])


@pytest.mark.trackml_1
def test_dask_one_event():
    df = dd.read_parquet(HITS_PARQUET, index='event_id', calculate_divisions=True) \
        .loc[1000].reset_index().compute()
    assert_frame_equal(_transform(df), get_one_event())


@pytest.mark.slow
@pytest.mark.trackml
@pytest.mark.trackml_1
def test_dask_sample():
    df = dd.read_parquet(HITS_PARQUET, index='event_id', calculate_divisions=True) \
             .loc[:1099].compute().reset_index()
    assert_frame_equal(_transform(df), get_sample())
