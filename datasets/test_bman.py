import pytest
from numpy.testing import assert_array_equal

from datasets.bman import get_hits_bman, get_hits_bman_one_event


@pytest.mark.bman
def test_get_hits_bman():
    hits = get_hits_bman()
    assert_array_equal(hits.index, range(15813216))
    assert_array_equal(hits.event_id.unique(), range(25000))
    assert set(hits.layer.unique()) == set(range(9))
    assert hits.track.min() == -1
    assert hits.track.dtype == 'int64'


def test_get_hits_bman_one_event():
    hits = get_hits_bman_one_event()
    assert_array_equal(hits.index, range(858))
    assert_array_equal(hits.event_id, [6] * 858)
    assert set(hits.layer.unique()) == set(range(9))
    assert hits.track.min() == -1
    assert hits.track.dtype == 'int64'
