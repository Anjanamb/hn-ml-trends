"""Trend detection: compare subtopic activity across time windows.

A subtopic is "trending" when its share of activity (story count or
total upvotes) in the current window is meaningfully higher than in
the previous window.

Two detectors share the same interface:
- ``frequency_deltas`` (weight_col=None): change in story-count share
- ``upvote_deltas``    (weight_col='score'): change in total-score share
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrendResult:
    category: str
    current_share: float
    previous_share: float
    delta: float          # current - previous, in share units (0..1)
    ratio: float          # current / previous (inf if previous == 0)


def _shares(df: pd.DataFrame, weight_col: str | None) -> pd.Series:
    """Return per-category share of the given weight column."""
    if weight_col is None:
        counts = df.groupby("category").size()
    else:
        counts = df.groupby("category")[weight_col].sum()
    total = counts.sum()
    if total <= 0:
        return counts * 0.0
    return counts / total


def category_deltas(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Compute per-category shares in each window and their delta.

    Both frames must have a ``category`` column and (if ``weight_col``
    is given) that column too. Returns a DataFrame sorted by absolute
    delta, descending.
    """
    cur = _shares(current, weight_col)
    prev = _shares(previous, weight_col)
    all_cats = sorted(set(cur.index) | set(prev.index))
    rows = []
    for c in all_cats:
        cur_s = float(cur.get(c, 0.0))
        prev_s = float(prev.get(c, 0.0))
        ratio = float("inf") if prev_s == 0 else cur_s / prev_s
        rows.append(TrendResult(c, cur_s, prev_s, cur_s - prev_s, ratio))
    out = pd.DataFrame([r.__dict__ for r in rows])
    return out.sort_values("delta", key=lambda s: s.abs(), ascending=False)


def top_trending(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    weight_col: str | None = None,
    top_n: int = 5,
    min_delta: float = 0.02,
) -> pd.DataFrame:
    """Categories whose share grew by at least ``min_delta`` in the
    current window, sorted by delta descending."""
    d = category_deltas(current, previous, weight_col)
    rising = d[d["delta"] >= min_delta].sort_values("delta", ascending=False)
    return rising.head(top_n)
