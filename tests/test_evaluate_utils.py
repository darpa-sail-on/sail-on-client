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


@pytest.fixture(scope="function")
def seeded_rng():
    """Fixture for providing seeded random number generator."""
    return np.random.default_rng(2022)


def test_check_novel_validity(seeded_rng):
    """Test for checking validity of novel and gt vector."""
    gt = np.random.randint(0, 2, size=(10))
    with pytest.raises(Exception):
        mismatched_p_novel = seeded_rng.standard_normal(11)
        check_novel_validity(mismatched_p_novel, gt)

    with pytest.raises(Exception):
        additional_dim = seeded_rng.standard_normal((10, 1))
        check_novel_validity(additional_dim, gt)

    with pytest.raises(Exception):
        invalid_p_novel = seeded_rng.standard_normal(10) * 10.0
        check_novel_validity(invalid_p_novel, gt)

    with pytest.raises(Exception):
        invalid_gt = seeded_rng.integers(0, 4, size=10)
        p_novel = seeded_rng.random(10)
        check_novel_validity(p_novel, invalid_gt)

    check_novel_validity(p_novel, gt)


def test_check_class_validity(seeded_rng):
    """Test for checking validity of novel and gt class matrix."""
    gt = seeded_rng.integers(0, 2, size=(10, 5))
    p_class = seeded_rng.random((10, 5))
    with pytest.raises(Exception):
        mismatched_p_class = seeded_rng.random((11, 5))
        check_class_validity(mismatched_p_class, gt)

    with pytest.raises(Exception):
        invalid_p_class = seeded_rng.random((10, 5)) * 10.0
        check_class_validity(invalid_p_class, gt)

    with pytest.raises(Exception):
        invalid_gt = seeded_rng.random(0, 10, size=(10, 5))
        check_class_validity(p_class, invalid_gt)

    check_class_validity(p_class, gt)


def test_top1_accuracy(seeded_rng):
    """Test top1 accuracy for novel and gt class matrix."""
    gt = seeded_rng.integers(low=0, high=6, size=100)
    p_class = seeded_rng.random((100, 6))
    acc = top1_accuracy(p_class, gt)
    assert acc == 0.17


def test_top3_accuracy(seeded_rng):
    """Test top3 accuracy for novel and gt class matrix."""
    gt = seeded_rng.integers(low=0, high=6, size=100)
    p_class = seeded_rng.random((100, 6))
    acc = top3_accuracy(p_class, gt)
    assert acc == 0.5


def test_get_rolling_stats(seeded_rng):
    """Test rolling stats calculation novel and gt class matrix."""
    gt = seeded_rng.integers(low=0, high=6, size=100)
    p_class = seeded_rng.random((100, 6))
    acc = get_rolling_stats(p_class, gt)
    assert all(
        map(
            lambda x: math.isclose(x[0], x[1], rel_tol=1e-3, abs_tol=1e-5),
            zip(acc, [0.18, 0.060927]),
        )
    )


def test_get_first_detect_novelty(seeded_rng):
    """Test first point where novelty is introduced based on novel and gt vector."""
    p_novel = seeded_rng.random((100))
    first_novelty = get_first_detect_novelty(p_novel, 0.5)
    assert first_novelty == 3
