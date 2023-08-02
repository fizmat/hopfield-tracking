import pytest

from datasets import get_hits


@pytest.mark.parametrize("dataset", ('simple', 'bman', 'spdsim', 'trackml', 'trackml_volume'))
def test_get_hits_one_event(dataset):
    events = get_hits(dataset, 1)
    assert len(events.event_id.unique()) == 1


@pytest.mark.parametrize("dataset",
                         ('simple', 'spdsim',
                          pytest.param('bman', marks=pytest.mark.bman),
                          pytest.param('trackml', marks=pytest.mark.trackml),
                          pytest.param('trackml_volume', marks=pytest.mark.trackml)
                          )
                         )
def test_get_hits_two_events(dataset):
    events = get_hits(dataset, 2)
    assert len(events.event_id.unique()) == 2
