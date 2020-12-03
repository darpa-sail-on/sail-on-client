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
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "domain": "image_classification",
            "test_ids": ["OND.1.1.1234"],
            "novelty_detector_class": "MockDetector",
        }
        config_name = "test_ond_config.json"
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield os.path.join(config_folder, config_name)


@pytest.fixture(scope="function")
def ond_config_with_feature_extraction():
    """Fixture to create a config file for feature extraction."""
    with TemporaryDirectory() as feature_dir:
        with TemporaryDirectory() as config_folder:
            dummy_config = {
                "domain": "image_classification",
                "test_ids": ["OND.1.1.1234"],
                "novelty_detector_class": "MockDetector",
                "feature_extraction_only": True,
                "save_features": True,
                "save_dir": feature_dir,
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
