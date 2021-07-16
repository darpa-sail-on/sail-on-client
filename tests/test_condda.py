"""Tests for CONDDA protocol."""

from tempfile import TemporaryDirectory
import pytest
import os

from sail_on_client.protocol.condda_protocol import Condda


@pytest.fixture(scope="function")
def condda_params():
    """Fixture to provide parameters for CONDDA Protocol."""
    with TemporaryDirectory() as save_dir:
        test_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(test_dir, "data")
        domain = "activity_recognition"
        seed = "5278"
        test_ids = ["CONDDA.10.90001.2100554"]
        yield dataset_root, domain, seed, test_ids, save_dir


@pytest.fixture(scope="function")
def condda_fe_params():
    """Fixture to create a config file for feature extraction."""
    feature_extraction_only = True
    save_features = True
    return feature_extraction_only, save_features


def test_initialize(
    server_setup, condda_params, condda_harness_instance, condda_algorithm_instance
):
    """
    Test condda protocol initialization.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        condda_params (tuple): Tuple to configure CONDDA parameters with all defaults
        condda_harness_instance: An instance of local harness
        condda_algorithm_instance: An instance of PreComputedONDAgent

    Return:
        None
    """
    algorithms = {"PreComputedCONDDAAgent": condda_algorithm_instance}
    dataset_root, domain, seed, test_ids, save_dir = condda_params
    Condda(
        algorithms,
        dataset_root,
        domain,
        condda_harness_instance,
        save_dir,
        seed,
        test_ids,
    )


def test_run_protocol(
    server_setup, condda_params, condda_harness_instance, condda_algorithm_instance
):
    """
    Test running protocol.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        condda_params (tuple): Tuple to configure CONDDA parameters with all defaults
        condda_harness_instance: An instance of local harness
        condda_algorithm_instance: An instance of PreComputedONDAgent

    Return:
        None
    """
    algorithms = {"PreComputedCONDDAAgent": condda_algorithm_instance}
    dataset_root, domain, seed, test_ids, save_dir = condda_params
    condda = Condda(
        algorithms,
        dataset_root,
        domain,
        condda_harness_instance,
        save_dir,
        seed,
        test_ids,
    )
    condda.run_protocol({})


def test_feature_extraction(
    server_setup,
    condda_params,
    condda_harness_instance,
    condda_algorithm_instance,
    condda_fe_params,
):
    """
    Test feature extraction.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        condda_params (tuple): Tuple to configure CONDDA parameters with all defaults
        condda_fe_params (tuple): Tuple to configure CONDDA parameters for feature extraction
        condda_harness_instance: An instance of local harness
        condda_algorithm_instance: An instance of PreComputedONDAgent

    Return:
        None
    """
    algorithms = {"PreComputedCONDDAAgent": condda_algorithm_instance}
    dataset_root, domain, seed, test_ids, save_dir = condda_params
    feature_extraction_only, save_features = condda_fe_params
    condda = Condda(
        algorithms,
        dataset_root,
        domain,
        condda_harness_instance,
        save_dir,
        seed,
        test_ids,
        feature_extraction_only=feature_extraction_only,
        save_features=save_features,
    )
    condda.run_protocol({})
