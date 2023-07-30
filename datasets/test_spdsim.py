import pytest
from numpy.testing import assert_array_equal

from datasets.spdsim import get_hits_spdsim, get_hits_spdsim_one_event


def test_get_hits_spdsim():
    hits = get_hits_spdsim(13, 7)
    assert_array_equal(hits.event_id.unique(), range(13))
    assert_array_equal(hits.columns, ['x', 'y', 'z', 'layer', 'track', 'event_id'])


def test_get_hits_spdsim_one_event():
    hits = get_hits_spdsim_one_event(7)
    assert_array_equal(hits.event_id, 0)
    assert_array_equal(hits.columns, ['x', 'y', 'z', 'layer', 'track', 'event_id'])


def test_spdsim_default_seed():
    assert_array_equal(get_hits_spdsim(), get_hits_spdsim())


def test_spdsim_same_seed():
    assert_array_equal(get_hits_spdsim(seed=13), get_hits_spdsim(seed=13))


def test_spdsim_different_seed():
    with pytest.raises(AssertionError):
        assert_array_equal(get_hits_spdsim(seed=13), get_hits_spdsim(seed=1))
