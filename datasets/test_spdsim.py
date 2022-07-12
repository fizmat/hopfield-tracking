from numpy.testing import assert_array_equal

from datasets.spdsim import get_hits_spdsim, get_hits_spdsim_one_event


def test_get_hits_spdsim():
    events = get_hits_spdsim(13, 7)
    assert_array_equal(events.event_id.unique(), range(13))
    assert_array_equal(events.columns, ['x', 'y', 'z', 'layer', 'track', 'event_id'])


def test_get_hits_spdsim_one_event():
    events = get_hits_spdsim_one_event(7)
    assert_array_equal(events.event_id, 0)
    assert_array_equal(events.columns, ['x', 'y', 'z', 'layer', 'track', 'event_id'])
