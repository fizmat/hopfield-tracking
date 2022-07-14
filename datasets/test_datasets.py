import pytest

from datasets import get_hits


def test_get_hits_one_event():
    for dataset in ('simple', 'bman', 'spdsim', 'trackml', 'trackml_volume'):
        events = get_hits(dataset, 1)
        assert len(events.event_id.unique()) == 1


@pytest.mark.trackml
@pytest.mark.bman
@pytest.mark.slow
def test_get_hits():
    for dataset in ('simple', 'bman', 'spdsim', 'trackml', 'trackml_volume'):
        events = get_hits(dataset, 2)
        assert len(events.event_id.unique()) == 2
