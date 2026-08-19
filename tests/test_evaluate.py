"""Tests for the classification / clustering metrics."""
from __future__ import annotations

import pytest

from src.evaluate import accuracy, cluster_purity, per_class_report


def test_accuracy_on_perfect_prediction():
    y = ["a", "b", "c", "a"]
    assert accuracy(y, y) == 1.0


def test_accuracy_on_all_wrong():
    assert accuracy(["a", "b", "c"], ["b", "c", "a"]) == 0.0


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_per_class_report_matches_hand_calculation():
    y_true = ["a", "a", "b", "b", "b", "c"]
    y_pred = ["a", "b", "b", "b", "c", "c"]
    r = per_class_report(y_true, y_pred, categories=["a", "b", "c"])

    # Class a: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=2/3
    assert r.loc["a", "support"] == 2
    assert r.loc["a", "precision"] == 1.0
    assert r.loc["a", "recall"] == 0.5
    assert abs(r.loc["a", "f1"] - (2 / 3)) < 1e-9

    # Class b: tp=2, fp=1, fn=1 -> P=2/3, R=2/3, F1=2/3
    assert r.loc["b", "support"] == 3
    assert abs(r.loc["b", "precision"] - (2 / 3)) < 1e-9
    assert abs(r.loc["b", "recall"] - (2 / 3)) < 1e-9

    # Class c: tp=1, fp=1, fn=0 -> P=0.5, R=1.0, F1=2/3
    assert r.loc["c", "support"] == 1
    assert r.loc["c", "precision"] == 0.5
    assert r.loc["c", "recall"] == 1.0


def test_cluster_purity_ideal_case():
    labels = ["a", "a", "b", "b"]
    clusters = [0, 0, 1, 1]
    assert cluster_purity(labels, clusters) == 1.0


def test_cluster_purity_ignores_noise_label_minus_one():
    labels =   ["a", "a", "b", "b", "a"]
    clusters = [0,   0,   1,   1,   -1]
    assert cluster_purity(labels, clusters) == 1.0


def test_cluster_purity_mixed_cluster():
    labels =   ["a", "a", "a", "b"]
    clusters = [0,   0,   1,   1]
    # cluster 0: 2 a (2 correct); cluster 1: 1 a + 1 b (1 correct)
    assert abs(cluster_purity(labels, clusters) - 0.75) < 1e-9
