"""Tests for precomputed detector."""

from sail_on_client.agent.pre_computed_detector import PreComputedONDAgent
from sail_on_client.agent.pre_computed_detector import PreComputedCONDDAAgent

import os
import pytest


@pytest.fixture(scope="function")
def precomputed_ond_agent():
    """Fixture for creating precomputed agent for OND."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    return PreComputedONDAgent("PreComputedONDAgent", cache_dir, False, 32)


@pytest.fixture(scope="function")
def precomputed_ond_agent_with_round():
    """Fixture for creating precomputed agent for OND with round file."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    return PreComputedONDAgent("PreComputedONDAgent", cache_dir, True, 32)


@pytest.fixture(scope="function")
def precomputed_ond_agent_with_features():
    """Fixture for creating precomputed detector instance with feature extraction."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    precomputed_ond_agent = PreComputedONDAgent(
        "PreComputedONDAgent", cache_dir, False, 32
    )
    precomputed_ond_agent.execute({"test_id": "OND.10.90001.2100554"}, "Initialize")
    dataset = os.path.join(
        test_dir, "data", "OND", "activity_recognition", "OND.10.90001.2100554.csv"
    )
    fe_toolset = {
        "dataset": dataset,
    }
    precomputed_ond_agent.execute(fe_toolset, "FeatureExtraction")
    return precomputed_ond_agent


@pytest.fixture(scope="function")
def precomputed_ond_agent_with_features_with_round():
    """Fixture for creating precomputed detector instance with feature extraction."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    precomputed_ond_agent = PreComputedONDAgent(
        "PreComputedONDAgent", cache_dir, True, 32
    )
    precomputed_ond_agent.execute({"test_id": "OND.10.90001.2100554"}, "Initialize")
    dataset = os.path.join(
        test_dir, "data", "OND", "activity_recognition", "OND.10.90001.2100554.csv"
    )
    fe_toolset = {
        "dataset": dataset,
    }
    precomputed_ond_agent.execute(fe_toolset, "FeatureExtraction")
    return precomputed_ond_agent


@pytest.fixture(scope="function")
def precomputed_condda_agent():
    """Fixture for creating precomputed agent for CONDDA."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    return PreComputedCONDDAAgent("PreComputedCONDDAAgent", cache_dir, False, 32)


@pytest.fixture(scope="function")
def precomputed_condda_agent_with_round():
    """Fixture for creating precomputed agent for CONDDA with round file."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    return PreComputedCONDDAAgent("PreComputedONDAgent", cache_dir, True, 32)


@pytest.fixture(scope="function")
def precomputed_condda_agent_with_features():
    """Fixture for creating precomputed detector instance with feature extraction."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    precomputed_condda_agent = PreComputedCONDDAAgent(
        "PreComputedCONDDAAgent", cache_dir, False, 32
    )
    precomputed_condda_agent.execute(
        {"test_id": "CONDDA.10.90001.2100554"}, "Initialize"
    )
    dataset = os.path.join(
        test_dir,
        "data",
        "CONDDA",
        "activity_recognition",
        "CONDDA.10.90001.2100554.csv",
    )
    fe_toolset = {
        "dataset": dataset,
    }
    precomputed_condda_agent.execute(fe_toolset, "FeatureExtraction")
    return precomputed_condda_agent


@pytest.fixture(scope="function")
def precomputed_condda_agent_with_features_with_round():
    """Fixture for creating precomputed detector instance with feature extraction."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    precomputed_condda_agent = PreComputedCONDDAAgent(
        "PreComputedCONDDAAgent", cache_dir, True, 32
    )
    precomputed_condda_agent.execute(
        {"test_id": "CONDDA.10.90001.2100554"}, "Initialize"
    )
    dataset = os.path.join(
        test_dir,
        "data",
        "CONDDA",
        "activity_recognition",
        "CONDDA.10.90001.2100554.csv",
    )
    fe_toolset = {
        "dataset": dataset,
    }
    precomputed_condda_agent.execute(fe_toolset, "FeatureExtraction")
    return precomputed_condda_agent


def test_ond_init():
    """
    Test precomputed detector init.

    Return:
        None
    """
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    precomputed_ond_agent = PreComputedONDAgent(
        "PreComputedONDAgent", cache_dir, False, 32
    )
    assert precomputed_ond_agent.algorithm_name == "PreComputedONDAgent"
    precomputed_ond_agent_with_round = PreComputedONDAgent(
        "PreComputedONDAgent", cache_dir, True, 32
    )
    assert precomputed_ond_agent_with_round.has_roundwise_file


def test_ond_initialize(precomputed_ond_agent, precomputed_ond_agent_with_round):
    """
    Test precomputed detector initialize step.

    Args:
        precomputed_ond_agent (PreComputedONDAgent): An instance of PreComputedONDAgent
        precomputed_ond_agent_with_round (PreComputedONDAgent): An instance of PreComputedONDAgent with roundwise file

    Return:
        None
    """
    precomputed_ond_agent.execute({"test_id": "OND.10.90001.2100554"}, "Initialize")
    precomputed_ond_agent_with_round.execute(
        {"test_id": "OND.10.90001.2100554"}, "Initialize"
    )


def test_ond_feature_extraction(precomputed_ond_agent):
    """
    Test precomputed detector feature extraction.

    Args:
        precomputed_ond_agent (PreComputedONDAgent): An instance of PreComputedONDAgent

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
    precomputed_ond_agent.execute(init_toolset, "Initialize")
    precomputed_ond_agent.execute(fe_toolset, "FeatureExtraction")


def test_ond_detection(
    precomputed_ond_agent_with_features, precomputed_ond_agent_with_features_with_round
):
    """
    Test precomputed detector detection.

    Args:
        precomputed_ond_agent_with_features (PreComputedONDAgent): An instance of PreComputedONDAgent
        precomputed_ond_agent_with_features_with_round (PreComputedONDAgent): An instance of PreComputedONDAgent

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_ond_agent_with_features.execute(toolset, "WorldDetection")
    toolset.update({"round_id": 1})
    precomputed_ond_agent_with_features_with_round.execute(toolset, "WorldDetection")


def test_ond_classification(
    precomputed_ond_agent_with_features, precomputed_ond_agent_with_features_with_round
):
    """
    Test precomputed detector classification.

    Args:
        precomputed_ond_agent_with_features (PreComputedONDAgent): An instance of PreComputedONDAgent
        precomputed_ond_agent_with_features_with_round (PreComputedONDAgent): An instance of PreComputedONDAgent

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_ond_agent_with_features.execute(toolset, "NoveltyClassification")
    toolset.update({"round_id": 1})
    precomputed_ond_agent_with_features_with_round.execute(
        toolset, "NoveltyClassification"
    )


def test_ond_characterization(precomputed_ond_agent_with_features):
    """
    Test precomputed detector characterization.

    Args:
        precomputed_ond_agent_with_features (PreComputedONDAgent): An instance of PreComputedONDAgent

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_ond_agent_with_features.execute(toolset, "NoveltyCharacterization")


def test_ond_adaption(precomputed_ond_agent_with_features):
    """
    Test precomputed detector adaption.

    Args:
        precomputed_ond_agent_with_features (PreComputedONDAgent): An instance of PreComputedONDAgent

    Return:
        None
    """
    toolset = {"test_id": "OND.10.90001.2100554"}
    precomputed_ond_agent_with_features.execute(toolset, "NoveltyAdaptation")


def test_condda_init():
    """
    Test precomputed detector in condda.

    Return:
        None
    """
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    precomputed_condda_agent = PreComputedCONDDAAgent(
        "PreComputedCONDDAAgent", cache_dir, False, 32
    )
    assert precomputed_condda_agent.algorithm_name == "PreComputedCONDDAAgent"
    precomputed_condda_agent_with_round = PreComputedCONDDAAgent(
        "PreComputedCONDDAAgent", cache_dir, True, 32
    )
    assert precomputed_condda_agent_with_round.has_roundwise_file


def test_condda_initialize(
    precomputed_condda_agent, precomputed_condda_agent_with_round
):
    """
    Test precomputed detector initialize step.

    Args:
        precomputed_condda_agent (PreComputedCONDDAAgent): An instance of PreComputedCONDDAAgent
        precomputed_condda_agent_with_round (PreComputedCONDDAAgent): An instance of PreComputedCONDDAAgent with roundwise file

    Return:
        None
    """
    precomputed_condda_agent.execute(
        {"test_id": "CONDDA.10.90001.2100554"}, "Initialize"
    )
    precomputed_condda_agent_with_round.execute(
        {"test_id": "CONDDA.10.90001.2100554"}, "Initialize"
    )


def test_condda_feature_extraction(precomputed_condda_agent):
    """
    Test precomputed detector feature extraction.

    Args:
        precomputed_condda_agent (PreComputedCONDDAAgent): An instance of PreComputedCONDDAAgent

    Return:
        None
    """
    test_dir = os.path.dirname(__file__)
    dataset = os.path.join(
        test_dir,
        "data",
        "CONDDA",
        "activity_recognition",
        "CONDDA.10.90001.2100554.csv",
    )
    fe_toolset = {
        "dataset": dataset,
    }
    init_toolset = {"test_id": "CONDDA.10.90001.2100554"}
    precomputed_condda_agent.execute(init_toolset, "Initialize")
    precomputed_condda_agent.execute(fe_toolset, "FeatureExtraction")


def test_condda_detection(
    precomputed_condda_agent_with_features,
    precomputed_condda_agent_with_features_with_round,
):
    """
    Test precomputed detector detection.

    Args:
        precomputed_condda_agent_with_features (PreComputedCONDDAAgent): An instance of PreComputedCONDDAAgent
        precomputed_condda_agent_with_features_with_round (PreComputedCONDDAAgent): An instance of PreComputedCONDDAAgent

    Return:
        None
    """
    toolset = {"test_id": "CONDDA.10.90001.2100554"}
    precomputed_condda_agent_with_features.execute(toolset, "WorldDetection")
    toolset.update({"round_id": 1})
    precomputed_condda_agent_with_features_with_round.execute(toolset, "WorldDetection")


def test_characterization(precomputed_condda_agent_with_features):
    """
    Test precomputed detector characterization.

    Args:
        precomputed_condda_agent_with_features (PreComputedCONDDAAgent): An instance of PreComputedCONDDAAgent

    Return:
        None
    """
    toolset = {"test_id": "CONDDA.10.90001.2100554"}
    precomputed_condda_agent_with_features.execute(toolset, "NoveltyCharacterization")
