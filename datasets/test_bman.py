import io
from itertools import zip_longest

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from datasets.bman import get_hits, get_one_event, ZIP_FILE, SCHEMA


@pytest.mark.bman
def test_get_hits_bman():
    hits = get_hits()
    assert_array_equal(hits.index, range(15813216))
    assert_array_equal(hits.event_id.unique(), range(25000))
    assert set(hits.layer.unique()) == set(range(9))
    assert hits.track.min() == -1
    assert hits.track.dtype == 'int16'


def test_get_hits_bman_one_event():
    hits = get_one_event()
    assert_array_equal(hits.index, range(858))
    assert_array_equal(hits.event_id, [6] * 858)
    assert set(hits.layer.unique()) == set(range(9))
    assert hits.track.min() == -1
    assert hits.track.dtype == 'int16'


@pytest.mark.bman
def test_schema():
    converted = pd.read_csv(ZIP_FILE, sep='\t', names=SCHEMA.keys(), dtype=SCHEMA, nrows=100_000)
    raw = pd.read_csv(ZIP_FILE, sep='\t', names=SCHEMA.keys(), nrows=100_000)
    c_buf = io.StringIO()
    r_buf = io.StringIO()
    raw.to_csv(r_buf)
    converted.to_csv(c_buf)
    r_buf.seek(0)
    c_buf.seek(0)
    for a, b in zip_longest(r_buf, c_buf):
        assert a == b
