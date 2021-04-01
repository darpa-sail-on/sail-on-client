"""Tests for CONDDA protocol."""

from tempfile import TemporaryDirectory
import json
import pytest
import os

from sail_on_client.protocol.condda import Condda
from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface


@pytest.fixture(scope="function")
def condda_config():
    """Fixture to create a temporal directory and create a json file in it."""
    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "data")
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "domain": "image_classification",
            "test_ids": ["CONDDA.1.1.1234"],
            "detectors": {
                "has_baseline": False,
                "has_reaction_baseline": False,
                "detector_configs": {"MockDetector": {}},
                "csv_folder": "",
            },
            "harness_config": {
                "url": "http://localhost:3307",
                "data_dir": f"{data_dir}",
            },
        }
        config_name = "test_condda_config.json"
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield os.path.join(config_folder, config_name)


@pytest.fixture(scope="function")
def condda_config_with_feature_extraction():
    """Fixture to create a config file for feature extraction."""
    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "data")
    with TemporaryDirectory() as feature_dir:
        with TemporaryDirectory() as config_folder:
            dummy_config = {
                "domain": "image_classification",
                "test_ids": ["CONDDA.1.1.1234"],
                "detectors": {
                    "has_baseline": False,
                    "has_reaction_baseline": False,
                    "detector_configs": {"MockDetector": {}},
                    "csv_folder": "",
                },
                "feature_extraction_only": True,
                "save_features": True,
                "save_dir": feature_dir,
                "harness_config": {
                    "url": "http://localhost:3307",
                    "data_dir": f"{data_dir}",
                },
            }
            config_name = "test_condda_config.json"
            json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
            yield os.path.join(config_folder, config_name)


def test_initialize(
    server_setup, get_interface_params, discoverable_plugins, condda_config
):
    """
    Test condda protocol initialization.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        discoverable_plugins (dict): Dictionary with the plugins
        condda_config (str): Path to json file

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    Condda(discoverable_plugins, "", par_interface, condda_config)
    local_interface = LocalInterface(config_name, config_directory)
    Condda(discoverable_plugins, "", local_interface, condda_config)


def test_run_protocol(
    server_setup, get_interface_params, discoverable_plugins, condda_config
):
    """
    Test running protocol.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        discoverable_plugins (dict): Dictionary with the plugins
        condda_config (str): Path to json file

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    condda = Condda(discoverable_plugins, "", par_interface, condda_config)
    condda.run_protocol()
    local_interface = LocalInterface(config_name, config_directory)
    condda = Condda(discoverable_plugins, "", local_interface, condda_config)
    condda.run_protocol()


def test_feature_extraction(
    server_setup,
    get_interface_params,
    discoverable_plugins,
    condda_config_with_feature_extraction,
):
    """
    Test feature extraction.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        discoverable_plugins (dict): Dictionary with the plugins
        condda_config_with_feature_extraction (str): Path to json file

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    condda = Condda(
        discoverable_plugins, "", par_interface, condda_config_with_feature_extraction
    )
    condda.run_protocol()
    local_interface = LocalInterface(config_name, config_directory)
    condda = Condda(
        discoverable_plugins, "", local_interface, condda_config_with_feature_extraction
    )
    condda.run_protocol()
