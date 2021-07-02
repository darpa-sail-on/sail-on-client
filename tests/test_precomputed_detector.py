"""Tests for precomputed detector."""

from sail_on_client.pre_computed_detector import PreComputedDetector

import os
import pytest


@pytest.fixture(scope="function")
def precomputed_detector():
    """Fixture for creating precomputed detector instance."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    init_toolset = {
        "cache_dir": cache_dir,
        "algorithm_name": "PreComputedDetector",
        "has_roundwise_file": False,
    }
    return PreComputedDetector(init_toolset)


@pytest.fixture(scope="function")
def precomputed_detector_with_round():
    """Fixture for creating precomputed detector instance with round file."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    init_toolset = {
        "cache_dir": cache_dir,
        "algorithm_name": "PreComputedDetector",
        "has_roundwise_file": True,
    }
    return PreComputedDetector(init_toolset)


@pytest.fixture(scope="function")
def precomputed_detector_with_features():
    """Fixture for creating precomputed detector instance with feature extraction."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    init_toolset = {
        "cache_dir": cache_dir,
        "algorithm_name": "PreComputedDetector",
        "has_roundwise_file": False,
    }
    precomputed_detector = PreComputedDetector(init_toolset)
    precomputed_detector.execute({"test_id": "OND.10.90001.2100554"}, "Initialize")
    dataset = os.path.join(
        test_dir, "data", "OND", "activity_recognition", "OND.10.90001.2100554.csv"
    )
    fe_toolset = {
        "dataset": dataset,
    }
    precomputed_detector.execute(fe_toolset, "FeatureExtraction")
    return precomputed_detector


@pytest.fixture(scope="function")
def precomputed_detector_with_features_with_round():
    """Fixture for creating precomputed detector instance with feature extraction."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    init_toolset = {
        "cache_dir": cache_dir,
        "algorithm_name": "PreComputedDetector",
        "has_roundwise_file": True,
    }
    precomputed_detector = PreComputedDetector(init_toolset)
    precomputed_detector.execute({"test_id": "OND.10.90001.2100554"}, "Initialize")
    dataset = os.path.join(
        test_dir, "data", "OND", "activity_recognition", "OND.10.90001.2100554.csv"
    )
    fe_toolset = {
        "dataset": dataset,
    }
    precomputed_detector.execute(fe_toolset, "FeatureExtraction")
    return precomputed_detector


def test_init():
    """
    Test precomputed detector init.

    Return:
        None
    """
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    init_toolset = {
        "cache_dir": cache_dir,
        "algorithm_name": "PreComputedDetector",
        "has_roundwise_file": False,
    }
    init_toolset_with_round = {
        "cache_dir": cache_dir,
        "algorithm_name": "PreComputedDetector",
        "has_roundwise_file": True,
    }
    precomputed_detector = PreComputedDetector(init_toolset)
    assert precomputed_detector.algorithm_name == "PreComputedDetector"
    precomputed_detector_with_round = PreComputedDetector(init_toolset_with_round)
    assert precomputed_detector_with_round.has_roundwise_file


def test_initialize(precomputed_detector, precomputed_detector_with_round):
    """
    Test precomputed detector initialize step.

    Args:
        precomputed_detector (PreComputedDetector): An instance of PreComputedDetector
        precomputed_detector_with_round (PreComputedDetector): An instance of PreComputedDetector with roundwise file

    Return:
        None
    """
    precomputed_detector.execute({"test_id": "OND.10.90001.2100554"}, "Initialize")
    precomputed_detector_with_round.execute({"test_id": "OND.10.90001.2100554"}, "Initialize")


def test_feature_extraction(precomputed_detector):
    """
    Test precomputed detector feature extraction.

    Args:
        precomputed_detector (PreComputedDetector): An instance of PreComputedDetector

    Return:
        None
    """
    test_dir = os.path.dirname(__file__)
    dataset = os.path.join(
        test_dir, "data", "OND", "activity_recognition", "OND.10.90001.2100554.csv"
    )
    fe_toolset = {
        "dataset": dataset,
    }
    init_toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_detector.execute(init_toolset, "Initialize")
    precomputed_detector.execute(fe_toolset, "FeatureExtraction")


def test_detection(
    precomputed_detector_with_features, precomputed_detector_with_features_with_round
):
    """
    Test precomputed detector detection.

    Args:
        precomputed_detector_with_features (PreComputedDetector): An instance of PreComputedDetector
        precomputed_detector_with_features_with_round (PreComputedDetector): An instance of PreComputedDetector

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_detector_with_features.execute(toolset, "WorldDetection")
    toolset.update({"round_id": 1})
    precomputed_detector_with_features_with_round.execute(toolset, "WorldDetection")


def test_classification(
    precomputed_detector_with_features, precomputed_detector_with_features_with_round
):
    """
    Test precomputed detector classification.

    Args:
        precomputed_detector_with_features (PreComputedDetector): An instance of PreComputedDetector
        precomputed_detector_with_features_with_round (PreComputedDetector): An instance of PreComputedDetector

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_detector_with_features.execute(toolset, "NoveltyClassification")
    toolset.update({"round_id": 1})
    precomputed_detector_with_features_with_round.execute(
        toolset, "NoveltyClassification"
    )


def test_characterization(precomputed_detector_with_features):
    """
    Test precomputed detector characterization.

    Args:
        precomputed_detector_with_features (PreComputedDetector): An instance of PreComputedDetector

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_detector_with_features.execute(toolset, "NoveltyCharacterization")


def test_adaption(precomputed_detector_with_features):
    """
    Test precomputed detector adaption.

    Args:
        precomputed_detector_with_features (PreComputedDetector): An instance of PreComputedDetector

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_detector_with_features.execute(toolset, "NoveltyAdaption")
