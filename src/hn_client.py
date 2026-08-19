"""Hacker News public API client.

Uses the Firebase-hosted read-only endpoints documented at
https://github.com/HackerNews/API. No authentication required, no
rate limits published, and no policy walls: it is literally a JSON
CDN.

Public helpers return plain dicts / DataFrames so notebooks and tests
do not need to touch the raw JSON structure.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
import requests


BASE_URL = "https://hacker-news.firebaseio.com/v0"

STORY_FIELDS = (
    "id", "time", "title", "text", "url", "by",
    "score", "descendants", "kids", "type", "dead", "deleted",
)


def top_story_ids(limit: int | None = None, timeout: float = 10.0) -> list[int]:
    """Return the current top-story IDs.

    HN exposes the top-500 list; pass ``limit`` to trim it.
    """
    r = requests.get(f"{BASE_URL}/topstories.json", timeout=timeout)
    r.raise_for_status()
    ids = r.json()
    return ids[:limit] if limit is not None else ids


def new_story_ids(limit: int | None = None, timeout: float = 10.0) -> list[int]:
    """Return the newest story IDs (chronological, newest first)."""
    r = requests.get(f"{BASE_URL}/newstories.json", timeout=timeout)
    r.raise_for_status()
    ids = r.json()
    return ids[:limit] if limit is not None else ids


def get_item(item_id: int, timeout: float = 10.0) -> dict:
    """Fetch a single HN item (story, comment, poll, ...) by ID."""
    r = requests.get(f"{BASE_URL}/item/{item_id}.json", timeout=timeout)
    r.raise_for_status()
    return r.json() or {}


def fetch_items(
    ids: Iterable[int],
    max_workers: int = 8,
    timeout: float = 10.0,
    sleep_between: float = 0.0,
) -> pd.DataFrame:
    """Fetch many items in parallel and return a tidy DataFrame.

    ``max_workers`` is deliberately modest (default 8) to be a polite
    API citizen; the Firebase endpoint has never announced a rate
    limit but there is no reason to hammer it.
    """
    ids = list(ids)
    rows: list[dict] = []

    def _one(i: int) -> dict:
        if sleep_between:
            time.sleep(sleep_between)
        return get_item(i, timeout=timeout)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, i): i for i in ids}
        for fut in as_completed(futures):
            item = fut.result()
            if item:
                rows.append(item)

    df = pd.DataFrame(rows)
    # Normalise: ensure every expected column exists.
    for col in STORY_FIELDS:
        if col not in df.columns:
            df[col] = None
    df = df[list(STORY_FIELDS)]

    if not df.empty:
        df["created_at"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["text"] = df["text"].fillna("")
        # Sort by id ascending so downstream consumers get a stable order.
        df = df.sort_values("id").reset_index(drop=True)
    return df


def fetch_top_comments(
    item_id: int,
    top_k: int = 3,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """For one story, fetch up to ``top_k`` top-level comments.

    HN does not expose a "sort by score" comment endpoint, so we fetch
    the first ``top_k`` non-deleted child comments in the ``kids`` list
    (which HN itself already orders by its own ranking).
    """
    parent = get_item(item_id, timeout=timeout)
    kids = parent.get("kids") or []
    rows = []
    for kid in kids[: top_k * 2]:   # small over-fetch to allow skipping deleted
        c = get_item(kid, timeout=timeout)
        if not c or c.get("deleted") or c.get("dead"):
            continue
        rows.append({
            "story_id":  item_id,
            "comment_id": c.get("id"),
            "by":        c.get("by"),
            "text":      c.get("text", ""),
            "time":      c.get("time"),
        })
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)
