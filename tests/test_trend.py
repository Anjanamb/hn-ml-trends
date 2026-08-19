"""Tests for the trend-detection helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from src.trend import category_deltas, top_trending


def _df(cats: list[str], weights: list[int] | None = None) -> pd.DataFrame:
    d = {"category": cats}
    if weights is not None:
        d["score"] = weights
    return pd.DataFrame(d)


def test_category_deltas_flat_no_change():
    cur = _df(["a", "a", "b", "b"])
    prev = _df(["a", "a", "b", "b"])
    d = category_deltas(cur, prev)
    assert set(d["category"]) == {"a", "b"}
    for _, row in d.iterrows():
        assert row["current_share"] == 0.5
        assert row["previous_share"] == 0.5
        assert row["delta"] == 0.0


def test_category_deltas_growth_in_one_category():
    cur = _df(["a", "a", "a", "b"])
    prev = _df(["a", "b", "b", "b"])
    d = category_deltas(cur, prev).set_index("category")
    assert d.loc["a", "delta"] == pytest.approx(0.5)
    assert d.loc["b", "delta"] == pytest.approx(-0.5)


def test_top_trending_returns_only_rising():
    cur = _df(["a", "a", "a", "a", "b"])
    prev = _df(["a", "b", "b", "b", "b"])
    out = top_trending(cur, prev, top_n=5, min_delta=0.1)
    assert list(out["category"]) == ["a"]


def test_category_deltas_weight_by_score():
    cur = _df(["a", "b"], weights=[100, 1])
    prev = _df(["a", "b"], weights=[1, 100])
    d = category_deltas(cur, prev, weight_col="score").set_index("category")
    assert d.loc["a", "current_share"] == pytest.approx(100 / 101)
    assert d.loc["b", "previous_share"] == pytest.approx(100 / 101)
