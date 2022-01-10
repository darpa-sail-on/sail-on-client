"""Tests for evaluate utils."""

import pytest
import numpy as np
import math

from sail_on_client.evaluate.utils import (
    check_novel_validity,
    check_class_validity,
    top1_accuracy,
    top3_accuracy,
    get_rolling_stats,
    get_first_detect_novelty,
)


def test_check_novel_validity():
    """Test for checking validity of novel and gt vector."""
    gt = np.random.randint(0, 2, size=(10))
    with pytest.raises(Exception):
        mismatched_p_novel = np.random.randn(11)
        check_novel_validity(mismatched_p_novel, gt)

    with pytest.raises(Exception):
        additional_dim = np.random.randn(10, 1)
        check_novel_validity(additional_dim, gt)

    with pytest.raises(Exception):
        invalid_p_novel = np.random.randn(10) * 10.0
        check_novel_validity(invalid_p_novel, gt)

    with pytest.raises(Exception):
        invalid_gt = np.random.randint(0, 4, size=(10))
        p_novel = np.random.rand(10)
        check_novel_validity(p_novel, invalid_gt)

    check_novel_validity(p_novel, gt)


def test_check_class_validity():
    """Test for checking validity of novel and gt class matrix."""
    gt = np.random.randint(0, 2, size=(10, 5))
    with pytest.raises(Exception):
        mismatched_p_class = np.random.rand(11, 5)
        check_class_validity(mismatched_p_class, gt)

    with pytest.raises(Exception):
        invalid_p_class = np.random.rand(10, 5) * 10.0
        check_class_validity(invalid_p_class, gt)

    with pytest.raises(Exception):
        invalid_gt = np.random.randint(0, 10, size=(10, 5))
        p_class = np.random.rand(10, 5)
        check_class_validity(p_class, invalid_gt)

    check_class_validity(p_class, gt)


def test_top1_accuracy():
    """Test top1 accuracy for novel and gt class matrix."""
    rng = np.random.default_rng(2022)
    gt = rng.integers(low=0, high=6, size=100)
    p_class = rng.random((100, 6))
    acc = top1_accuracy(p_class, gt)
    assert acc == 0.17


def test_top3_accuracy():
    """Test top3 accuracy for novel and gt class matrix."""
    rng = np.random.default_rng(2022)
    gt = rng.integers(low=0, high=6, size=100)
    p_class = rng.random((100, 6))
    acc = top3_accuracy(p_class, gt)
    assert acc == 0.5


def test_get_rolling_stats():
    """Test rolling stats calculation novel and gt class matrix."""
    rng = np.random.default_rng(2022)
    gt = rng.integers(low=0, high=6, size=100)
    p_class = rng.random((100, 6))
    acc = get_rolling_stats(p_class, gt)
    assert all(
        map(
            lambda x: math.isclose(x[0], x[1], rel_tol=1e-3, abs_tol=1e-5),
            zip(acc, [0.18, 0.060927]),
        )
    )


def test_get_first_detect_novelty():
    """Test first point where novelty is introduced based on novel and gt vector."""
    rng = np.random.default_rng(2022)
    p_novel = rng.random((100))
    first_novelty = get_first_detect_novelty(p_novel, 0.5)
    assert first_novelty == 3
