"""Tests for the label registry and gold-set I/O."""
from __future__ import annotations

import pandas as pd
import pytest

from src.labels import CATEGORIES, category_names, load_gold, prompt_snippet, save_gold


def test_category_names_are_non_empty_and_unique():
    names = category_names()
    assert len(names) >= 3
    assert len(set(names)) == len(names)


def test_prompt_snippet_lists_every_category():
    snippet = prompt_snippet()
    for name in CATEGORIES:
        assert name in snippet


def test_load_gold_round_trip(tmp_path):
    df = pd.DataFrame({
        "id":       [111, 222, 333],
        "category": ["research", "tool", "tutorial"],
    })
    p = tmp_path / "gold.csv"
    save_gold(df, p)
    loaded = load_gold(p)
    assert loaded.index.name == "id"
    assert loaded.loc[111, "category"] == "research"
    assert set(loaded["category"]) == {"research", "tool", "tutorial"}


def test_load_gold_rejects_unknown_labels(tmp_path):
    df = pd.DataFrame({"id": [1], "category": ["not_a_real_label"]})
    p = tmp_path / "bad.csv"
    save_gold(df, p)
    with pytest.raises(ValueError):
        load_gold(p)


def test_load_gold_rejects_missing_columns(tmp_path):
    df = pd.DataFrame({"id": [1], "wrong_col": ["research"]})
    p = tmp_path / "bad.csv"
    save_gold(df, p)
    with pytest.raises(ValueError):
        load_gold(p)
