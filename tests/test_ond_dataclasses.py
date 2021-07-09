"""Tests for OND dataclasses."""

import pytest
import uuid
from sail_on_client.mock import MockDetector
from sail_on_client.protocol.ond_dataclasses import (
    AlgorithmAttributes,
    InitializeParams,
    NoveltyClassificationParams,
    NoveltyAdaptationParams,
    NoveltyCharacterizationParams,
)


@pytest.fixture(scope="function")
def algorithm_attributes_params():
    """Fixture for providing parameters for initializing AlgorithmAttributes."""
    return {
        "name": "MockDetector",
        "detection_threshold": 0.5,
        "instance": MockDetector({}),
        "is_baseline": False,
        "is_reaction_baseline": False,
        "package_name": "sail-on-client",
        "parameters": {"param1": "foo", "param2": "bar"},
        "session_id": str(uuid.uuid4()),
        "test_ids": ["OND.10.90001.2100554", "OND.9.90001.2100554"],
    }


@pytest.fixture(scope="function")
def initialize_params():
    """Fixture for providing parameters for initialize InitializeParams."""
    return {
        "parameters": {"param1": "foo", "param2": "bar"},
        "session_id": str(uuid.uuid4()),
        "test_id": "OND.10.90001.2100554",
        "feedback_instance": None,
    }


@pytest.fixture(scope="function")
def novelty_classification_params():
    """Fixture for providing parameters for initialize NoveltyClassificationParams."""
    return {"features_dict": {}, "logit_dict": {}, "round_id": 0}


@pytest.fixture(scope="function")
def novelty_adaptation_params():
    """Fixture for providing parameters for initialize NoveltyAdaptationParams."""
    return {"round_id": 0}


@pytest.fixture(scope="function")
def novelty_characterization_params():
    """Fixture for providing parameters for initialize NoveltyCharacterizationParams."""
    return {"dataset_ids": []}


def test_algorithm_attributes_initialize(algorithm_attributes_params):
    """
    Test AlgorithmAttributes __init__.

    Args:
        algorithm_attributes_params: Dictionary with parameters to initialize AlgorithmAttributes

    Returns:
        None
    """
    AlgorithmAttributes(**algorithm_attributes_params)


def test_algorithm_attributes_named_version(algorithm_attributes_params):
    """
    Test AlgorithmAttributes named version.

    Args:
        algorithm_attributes_params: Dictionary with parameters to initialize AlgorithmAttributes

    Returns:
        None
    """
    algorithm_attributes = AlgorithmAttributes(**algorithm_attributes_params)
    assert "MockDetector" in algorithm_attributes.named_version()


def test_algorithm_attributes_removed_completed_tests(algorithm_attributes_params):
    """
    Test AlgorithmAttributes remove completed tests.

    Args:
        algorithm_attributes_params: Dictionary with parameters to initialize AlgorithmAttributes

    Returns:
        None
    """
    algorithm_attributes = AlgorithmAttributes(**algorithm_attributes_params)
    assert algorithm_attributes.test_ids == [
        "OND.10.90001.2100554",
        "OND.9.90001.2100554",
    ]
    algorithm_attributes.remove_completed_tests(["OND.9.90001.2100554"])
    assert algorithm_attributes.test_ids == ["OND.10.90001.2100554"]


def test_algorithm_attributes_merge_detector_params(algorithm_attributes_params):
    """
    Test AlgorithmAttributes remove completed tests.

    Args:
        algorithm_attributes_params: Dictionary with parameters to initialize AlgorithmAttributes

    Returns:
        None
    """
    algorithm_attributes = AlgorithmAttributes(**algorithm_attributes_params)
    assert algorithm_attributes.parameters == {"param1": "foo", "param2": "bar"}
    algorithm_attributes.merge_detector_params({"param1": "bar"})
    assert algorithm_attributes.parameters == {"param1": "bar", "param2": "bar"}
    algorithm_attributes.merge_detector_params(
        {"param1": "foo", "param2": "foo"}, exclude_keys=["param2"]
    )
    assert algorithm_attributes.parameters == {"param1": "foo", "param2": "bar"}


def test_initialize_params_initialize(initialize_params):
    """
    Test InitializeParams __init__.

    Args:
        initialize_params: Dictionary with parameters to initialize InitializeParams

    Returns:
        None
    """
    InitializeParams(**initialize_params)


def test_initialize_params_get_toolset(initialize_params):
    """
    Test InitializeParams get_toolset.

    Args:
        initialize_params: Dictionary with parameters to initialize InitializeParams

    Returns:
        None
    """
    init_params = InitializeParams(**initialize_params)
    assert init_params.get_toolset() == {
        "param1": "foo",
        "param2": "bar",
        "session_id": initialize_params["session_id"],
        "test_id": "OND.10.90001.2100554",
        "test_type": "",
    }


def test_novelty_classification_initialize(novelty_classification_params):
    """
    Test FeatureExtractionParams __init__.

    Args:
        novelty_classification_params: Dictionary with parameters to initialize
                                       NoveltyClassificationParams

    Returns:
        None
    """
    NoveltyClassificationParams(**novelty_classification_params)


def test_novelty_classification_params_get_toolset(novelty_classification_params):
    """
    Test FeatureExtractionParams get_toolset.

    Args:
        novelty_classification_params: Dictionary with parameters to initialize
                                       NoveltyClassificationParams

    Returns:
        None
    """
    nc_params = NoveltyClassificationParams(**novelty_classification_params)
    assert nc_params.get_toolset() == {
        "features_dict": {},
        "logit_dict": {},
        "round_id": 0,
    }


def test_novelty_adaptation_initialize(novelty_adaptation_params):
    """
    Test NoveltyAdaptationParams __init__.

    Args:
        novelty_adaptation_params: Dictionary with parameters to initialize NoveltyAdaptationParams

    Returns:
        None
    """
    NoveltyAdaptationParams(**novelty_adaptation_params)


def test_novelty_adaptation_params_get_toolset(novelty_adaptation_params):
    """
    Test NoveltyAdaptationParams get_toolset.

    Args:
        novelty_adaptation_params: Dictionary with parameters to initialize NoveltyAdaptationParams

    Returns:
        None
    """
    na_params = NoveltyAdaptationParams(**novelty_adaptation_params)
    assert na_params.get_toolset() == {"round_id": 0}


def test_novelty_characterization_initialize(novelty_characterization_params):
    """
    Test NoveltyAdaptationParams __init__.

    Args:
        novelty_characterization_params: Dictionary with parameters to initialize NoveltyCharacterizationParams

    Returns:
        None
    """
    NoveltyCharacterizationParams(**novelty_characterization_params)


def test_novelty_characterization_params_get_toolset(novelty_characterization_params):
    """
    Test NoveltyCharacterizationParams get_toolset.

    Args:
        novelty_characterization_params: Dictionary with parameters to initialize NoveltyCharacterizationParams

    Returns:
        None
    """
    nc_params = NoveltyCharacterizationParams(**novelty_characterization_params)
    assert nc_params.get_toolset() == {"dataset_ids": []}
