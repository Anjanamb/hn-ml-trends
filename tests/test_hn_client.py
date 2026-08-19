"""Tests for the Hacker News client.

Uses ``requests-mock``-style monkeypatching so tests never hit the
live HN endpoint (avoids flakiness + keeps CI offline-safe).
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd

from src import hn_client


def _mock_response(json_data):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json = MagicMock(return_value=json_data)
    return m


def test_top_story_ids_returns_first_n():
    fake_ids = list(range(1000, 2000))
    with patch("src.hn_client.requests.get", return_value=_mock_response(fake_ids)):
        got = hn_client.top_story_ids(limit=5)
    assert got == [1000, 1001, 1002, 1003, 1004]


def test_top_story_ids_no_limit_returns_all():
    fake_ids = [1, 2, 3]
    with patch("src.hn_client.requests.get", return_value=_mock_response(fake_ids)):
        got = hn_client.top_story_ids()
    assert got == fake_ids


def test_get_item_returns_dict():
    fake_item = {"id": 42, "title": "hello", "type": "story", "score": 10}
    with patch("src.hn_client.requests.get", return_value=_mock_response(fake_item)):
        got = hn_client.get_item(42)
    assert got["id"] == 42
    assert got["title"] == "hello"


def test_get_item_handles_null_response():
    with patch("src.hn_client.requests.get", return_value=_mock_response(None)):
        got = hn_client.get_item(999999999)
    assert got == {}


def test_fetch_items_returns_tidy_dataframe():
    items = {
        100: {"id": 100, "title": "A", "type": "story", "score": 5,
              "time": 1_700_000_000, "by": "u1", "descendants": 2},
        101: {"id": 101, "title": "B", "type": "story", "score": 8,
              "time": 1_700_003_600, "by": "u2", "descendants": 0},
        102: {"id": 102, "title": "C", "type": "story", "score": 1,
              "time": 1_700_007_200, "by": "u3", "descendants": 5,
              "text": "some body text"},
    }

    def fake_get(url, timeout=None):
        item_id = int(url.rsplit("/", 1)[-1].removesuffix(".json"))
        return _mock_response(items[item_id])

    with patch("src.hn_client.requests.get", side_effect=fake_get):
        df = hn_client.fetch_items([100, 101, 102])

    assert len(df) == 3
    # Every expected field present
    for col in hn_client.STORY_FIELDS:
        assert col in df.columns
    # created_at attached
    assert "created_at" in df.columns
    # text NaN was filled to empty string
    assert df.loc[df["id"] == 100, "text"].iloc[0] == ""
    # Ordering by id ascending
    assert list(df["id"]) == [100, 101, 102]


def test_fetch_items_empty_input():
    with patch("src.hn_client.requests.get") as m:
        df = hn_client.fetch_items([])
    m.assert_not_called()
    assert len(df) == 0
    # Still returns all expected columns
    for col in hn_client.STORY_FIELDS:
        assert col in df.columns
