"""Tests for OND protocol."""

from tempfile import TemporaryDirectory
import pytest
import os

from sail_on_client.protocol.ond_protocol import ONDProtocol
from sail_on_client.agent.pre_computed_detector import PreComputedONDAgent


@pytest.fixture(scope="function")
def ond_params():
    """Fixture to create a provide parameters for OND Protocol."""
    with TemporaryDirectory() as save_dir:
        test_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(test_dir, "data")
        domain = "activity_recognition"
        seed = "5278"
        test_ids = ["OND.10.90001.2100554"]
        yield dataset_root, domain, seed, test_ids, save_dir


@pytest.fixture(scope="function")
def ond_fe_params():
    """Fixture to create a config file for feature extraction."""
    feature_extraction_only = True
    save_features = True
    return feature_extraction_only, save_features


@pytest.fixture
def ond_reaction_baseline_params():
    """Fixture to create a reaction baseline."""
    baseline_class = "BaselinePreComputedONDDetector"
    has_reaction_baseline = True
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    baseline_algorithm = {baseline_class: PreComputedONDAgent("BaselinePreComputedONDAgent",
                                                              cache_dir, False, 32)}
    return baseline_class, has_reaction_baseline, baseline_algorithm


def test_initialize(
    server_setup, ond_params, ond_harness_instance, ond_algorithm_instance
):
    """
    Test ond protocol initialization.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        ond_params (tuple): Tuple to configure OND parameters with all defaults
        ond_harness_instance: An instance of local harness
        ond_algorithm_instance: An instance of PreComputedONDAgent

    Return:
        None
    """
    algorithms = {"PreComputedONDAgent": ond_algorithm_instance}
    dataset_root, domain, seed, test_ids, save_dir = ond_params
    ONDProtocol(algorithms, dataset_root, domain, ond_harness_instance, save_dir,
                seed, test_ids)


def test_run_protocol(
    server_setup, ond_params, ond_harness_instance, ond_algorithm_instance
):
    """
    Test running protocol.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        ond_params (tuple): Tuple to configure OND parameters with all defaults
        ond_harness_instance: An instance of local harness
        ond_algorithm_instance: An instance of PreComputedONDAgent

    Return:
        None
    """
    algorithms = {"PreComputedONDAgent": ond_algorithm_instance}
    dataset_root, domain, seed, test_ids, save_dir = ond_params
    ond = ONDProtocol(algorithms, dataset_root, domain, ond_harness_instance,
                      save_dir, seed, test_ids)
    ond.run_protocol({})


def test_feature_extraction(
    server_setup, ond_params, ond_fe_params, ond_harness_instance, ond_algorithm_instance
):
    """
    Test feature extraction only.
        server_setup (tuple): Tuple containing url and result directory
        ond_params (tuple): Tuple to configure OND parameters with all defaults
        ond_fe_params (tuple): Tuple to configure OND parameters with feature extraction
        ond_harness_instance: An instance of local harness
        ond_algorithm_instance: An instance of PreComputedONDAgent

    Args:

    Return:
        None
    """
    algorithms = {"PreComputedONDAgent": ond_algorithm_instance}
    dataset_root, domain, seed, test_ids, save_dir = ond_params
    feature_extraction_only, save_features = ond_fe_params
    ond = ONDProtocol(algorithms, dataset_root, domain, ond_harness_instance,
                      save_dir, seed, test_ids,
                      feature_extraction_only=feature_extraction_only,
                      save_features=save_features)
    ond.run_protocol({})


def test_reaction_baseline(
    server_setup, ond_params, ond_reaction_baseline_params, ond_harness_instance, ond_algorithm_instance
):
    """
    Test reaction baseline with a detector.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        ond_params (tuple): Tuple to configure OND parameters with all defaults
        ond_reaction_baseline_params (tuple): Tuple for testing with baseline
        ond_harness_instance: An instance of local harness
        ond_algorithm_instance: An instance of PreComputedONDAgent

    Return:
        None
    """
    algorithms = {"PreComputedONDAgent": ond_algorithm_instance}
    dataset_root, domain, seed, test_ids, save_dir = ond_params
    baseline_class, has_reaction_baseline, baseline_algorithm = ond_reaction_baseline_params
    algorithms.update(baseline_algorithm)
    ond = ONDProtocol(algorithms, dataset_root, domain, ond_harness_instance,
                      save_dir, seed, test_ids, baseline_class=baseline_class,
                      has_reaction_baseline=has_reaction_baseline,
                      is_eval_enabled=True, is_eval_roundwise_enabled=True)
    ond.run_protocol({})
