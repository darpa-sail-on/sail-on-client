"""Tests for visual dataclasses."""

import pytest
from sail_on_client.protocol.visual_dataclasses import (
    FeatureExtractionParams,
    WorldChangeDetectionParams,
)


@pytest.fixture(scope="function")
def feature_extraction_params():
    """Fixture for providing parameters for initialize FeatureExtractionParams."""
    return {"dataset": "", "data_root": "", "round_id": 0}


@pytest.fixture(scope="function")
def world_change_detection_params():
    """Fixture for providing parameters for initialize WorldChangeDetectionParams."""
    return {
        "features_dict": {},
        "logit_dict": {},
        "redlight_image": "test.png",
        "round_id": 0,
    }


def test_feature_extraction_initialize(feature_extraction_params):
    """
    Test FeatureExtractionParams __init__.

    Args:
        feature_extraction_params: Dictionary with parameters to initialize FeatureExtractionParams

    Returns:
        None
    """
    FeatureExtractionParams(**feature_extraction_params)


def test_feature_extraction_params_get_toolset(feature_extraction_params):
    """
    Test FeatureExtractionParams get_toolset.

    Args:
        feature_extraction_params: Dictionary with parameters to initialize FeatureExtractionParams

    Returns:
        None
    """
    fe_params = FeatureExtractionParams(**feature_extraction_params)
    assert fe_params.get_toolset() == {
        "dataset": "",
        "dataset_root": "",
        "round_id": 0,
    }


def test_world_change_detection_initialize(world_change_detection_params):
    """
    Test WorldChangeDetectionParams __init__.

    Args:
        world_change_detection_params: Dictionary with parameters to initialize WorldChangeDetectionParams

    Returns:
        None
    """
    WorldChangeDetectionParams(**world_change_detection_params)


def test_world_change_detection_get_toolset(world_change_detection_params):
    """
    Test WorldChangeDetectionParams get_toolset.

    Args:
        world_change_detection_params: Dictionary with parameters to initialize WorldChangeDetectionParams

    Returns:
        None
    """
    wc_params = WorldChangeDetectionParams(**world_change_detection_params)
    assert wc_params.get_toolset() == {
        "features_dict": {},
        "logit_dict": {},
        "round_id": 0,
        "redlight_image": "test.png",
    }
