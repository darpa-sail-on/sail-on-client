"""Tests for CONDDA protocol."""

from tempfile import TemporaryDirectory
import pytest
import os

from sail_on_client.protocol.condda_protocol import Condda
from sail_on_client.harness.local_harness import LocalHarness


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


@pytest.fixture(scope="function")
def condda_protocol_cfg_params():
    """Fixture for creating algorithm cfg."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    data_dir = os.path.join(test_dir, "data")
    gt_dir = os.path.join(data_dir, "CONDDA", "activity_recognition")
    gt_config = os.path.join(
        data_dir, "OND", "activity_recognition", "activity_recognition.json"
    )
    algorithms = {
        "algorithms": {
            "PreComputedCONDDAAgent": {
                "class": "PreComputedCONDDAAgent",
                "config": {
                    "algorithm_name": "PreComputedCONDDAAgent",
                    "cache_dir": cache_dir,
                    "has_roundwise_file": False,
                    "round_size": 32,
                },
            }
        },
        "harness": {
            "class": "LocalHarness",
            "config": {"data_dir": data_dir, "gt_dir": gt_dir, "gt_config": gt_config},
        },
    }
    return algorithms


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


def test_condda_from_config(condda_params, condda_protocol_cfg_params):
    """
    Test from_config in ond_protocol.

    Args:
        condda_params (tuple): Tuple to configure CONDDA parameters with all defaults
        condda_protocol_cfg_params: dictionary with the algorithms

    Returns:
        None
    """
    dataset_root, domain, seed, test_ids, save_dir = condda_params
    condda_protocol_cfg_params.update(
        {
            "dataset_root": dataset_root,
            "domain": domain,
            "seed": seed,
            "test_ids": test_ids,
            "save_dir": save_dir,
        }
    )
    condda_protocol = Condda.from_config(condda_protocol_cfg_params)
    assert "PreComputedCONDDAAgent" in condda_protocol.algorithms.keys()
    assert isinstance(condda_protocol.harness, LocalHarness)


def test_condda_get_config(condda_params, condda_protocol_cfg_params):
    """
    Test from_config in condda_protocol.

    Args:
        condda_params (tuple): Tuple to configure CONDDA parameters with all defaults
        condda_protocol_cfg_params: dictionary with the algorithms

    Returns:
        None
    """
    dataset_root, domain, seed, test_ids, save_dir = condda_params
    condda_protocol_cfg_params.update(
        {
            "dataset_root": dataset_root,
            "domain": domain,
            "seed": seed,
            "test_ids": test_ids,
            "save_dir": save_dir,
        }
    )
    condda_protocol = Condda.from_config(condda_protocol_cfg_params)
    condda_config = condda_protocol.get_config()
    # Testing the validity of the config by reconstructing the protocol from it
    Condda.from_config(condda_config)


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
