"""Tests for OND protocol."""

from tempfile import TemporaryDirectory
import json
import pytest
import os

from sail_on_client.protocol.ond_protocol import SailOn
from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface


@pytest.fixture(scope="function")
def ond_config():
    """Fixture to create a temporal directory and create a json file in it."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    data_dir = os.path.join(test_dir, "data")
    gt_dir = os.path.join(data_dir, "OND", "activity_recognition")
    gt_config = os.path.join(gt_dir, "activity_recognition")
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "domain": "activity_recognition",
            "test_ids": ["OND.10.90001.2100554"],
            "detectors": {
                "has_baseline": False,
                "has_reaction_baseline": False,
                "detector_configs": {
                    "PreComputedDetector": {
                        "cache_dir": cache_dir,
                        "algorithm_name": "PreComputedDetector",
                        "has_roundwise_file": False,
                    }
                },
                "csv_folder": "",
            },
            "harness_config": {
                "url": "http://localhost:3307",
                "data_dir": f"{data_dir}",
                "gt_dir": f"{gt_dir}",
                "gt_config": f"{gt_config}",
            },
        }
        config_name = "test_ond_config.json"
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield os.path.join(config_folder, config_name)


@pytest.fixture(scope="function")
def ond_config_with_feature_extraction():
    """Fixture to create a config file for feature extraction."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    data_dir = os.path.join(test_dir, "data")
    gt_dir = os.path.join(data_dir, "OND", "activity_recognition")
    gt_config = os.path.join(gt_dir, "activity_recognition.json")
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "domain": "activity_recognition",
            "test_ids": ["OND.10.90001.2100554"],
            "detectors": {
                "has_baseline": False,
                "has_reaction_baseline": False,
                "detector_configs": {
                    "PreComputedDetector": {
                        "cache_dir": cache_dir,
                        "algorithm_name": "PreComputedDetector",
                        "has_roundwise_file": False,
                    }
                },
                "csv_folder": "",
            },
            "harness_config": {
                "url": "http://localhost:3307",
                "data_dir": f"{data_dir}",
                "gt_dir": f"{gt_dir}",
                "gt_config": f"{gt_config}",
            },
        }
        config_name = "test_ond_config.json"
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield os.path.join(config_folder, config_name)


@pytest.fixture(scope="function")
def ond_config_with_reaction_baseline():
    """Fixture to create a reaction baseline."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    data_dir = os.path.join(test_dir, "data")
    gt_dir = os.path.join(data_dir, "OND", "activity_recognition")
    gt_config = os.path.join(gt_dir, "activity_recognition.json")
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "domain": "activity_recognition",
            "test_ids": ["OND.10.90001.2100554"],
            "detectors": {
                "has_baseline": False,
                "has_reaction_baseline": True,
                "baseline_class": "BaselinePreComputedDetector",
                "csv_folder": "",
                "detector_configs": {
                    "PreComputedDetector": {
                        "cache_dir": cache_dir,
                        "algorithm_name": "PreComputedDetector",
                        "has_roundwise_file": False,
                    },
                    "BaselinePreComputedDetector": {
                        "cache_dir": cache_dir,
                        "algorithm_name": "BaselinePreComputedDetector",
                        "has_roundwise_file": False,
                    },
                },
            },
            "harness_config": {
                "url": "http://localhost:3307",
                "data_dir": f"{data_dir}",
                "gt_dir": f"{gt_dir}",
                "gt_config": f"{gt_config}",
            },
        }
        config_name = "test_ond_config.json"
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield os.path.join(config_folder, config_name)


def test_initialize(
    server_setup, get_interface_params, discoverable_plugins, ond_config
):
    """
    Test ond protocol initialization.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        discoverable_plugins (dict): Dictionary with the plugins
        ond_config (str): Path to json file

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    SailOn(discoverable_plugins, "", par_interface, ond_config)
    local_interface = LocalInterface(config_name, config_directory)
    SailOn(discoverable_plugins, "", local_interface, ond_config)


def test_run_protocol(
    server_setup, get_interface_params, discoverable_plugins, ond_config
):
    """
    Test running protocol.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        discoverable_plugins (dict): Dictionary with the plugins
        ond_config (str): Path to json file

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    ond = SailOn(discoverable_plugins, "", par_interface, ond_config)
    ond.run_protocol()
    local_interface = LocalInterface(config_name, config_directory)
    SailOn(discoverable_plugins, "", local_interface, ond_config)
    ond.run_protocol()


def test_feature_extraction(
    server_setup,
    get_interface_params,
    discoverable_plugins,
    ond_config_with_feature_extraction,
):
    """
    Test feature extraction.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        discoverable_plugins (dict): Dictionary with the plugins
        ond_config_with_feature_extraction (str): Path to json file

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    ond = SailOn(
        discoverable_plugins, "", par_interface, ond_config_with_feature_extraction
    )
    ond.run_protocol()
    local_interface = LocalInterface(config_name, config_directory)
    SailOn(
        discoverable_plugins, "", local_interface, ond_config_with_feature_extraction
    )
    ond.run_protocol()


def test_reaction_baseline(
    server_setup,
    get_ar_interface_params,
    discoverable_plugins,
    ond_config_with_reaction_baseline,
):
    """
    Test reaction baseline with a detector.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_ar_interface_params (tuple): Tuple to configure par interface
        discoverable_plugins (dict): Dictionary with the plugins
        ond_config_with_reaction_baseline(str): Path to json file

    Return:
        None
    """
    config_directory, config_name = get_ar_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    ond = SailOn(
        discoverable_plugins, "", local_interface, ond_config_with_reaction_baseline
    )
    ond.run_protocol()
