"""Helpers for Tests."""

import pytest
import os
import time
import shutil
import json
import ubelt as ub
import multiprocessing
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from pkg_resources import iter_entry_points
from pkg_resources import DistributionNotFound

from sail_on.api import server
from sail_on.api.file_provider import FileProvider

log = logging.getLogger(__name__)

URL = "http://localhost:3307"
CONFIG_NAME = "configuration.json"


@pytest.fixture(scope="function")
def server_setup():
    """Fixture to setup server and remove artifacts associated with the server after the test."""
    data_dir = Path(f"{os.path.dirname(__file__)}/data")
    result_dir = Path(f"{os.path.dirname(__file__)}/server_results_{time.time()}")
    ub.ensuredir(data_dir)
    ub.ensuredir(result_dir)

    url = URL
    server.set_provider(FileProvider(data_dir, result_dir))
    api_process = multiprocessing.Process(target=server.init, args=("localhost", 3307))
    api_process.start()
    yield url, result_dir
    api_process.terminate()
    api_process.join()
    shutil.rmtree(result_dir)


@pytest.fixture(scope="function")
def get_interface_params():
    """Fixture to create a temporal directory and add a configuration.json in it."""
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "url": URL,
            "data_dir": f"{os.path.dirname(__file__)}/data",
            "gt_dir": f"{os.path.dirname(__file__)}/data/OND/image_classification",
            "gt_config": f"{os.path.dirname(__file__)}/data/OND/image_classification/image_classification.json",
        }
        config_name = "configuration.json"
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield config_folder, config_name


@pytest.fixture(scope="function")
def get_ic_interface_params():
    """Fixture to create a temporary directory and add a configuration.json in it."""
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "url": URL,
            "data_dir": f"{os.path.dirname(__file__)}/data",
            "gt_dir": f"{os.path.dirname(__file__)}/data/OND/image_classification",
            "gt_config": f"{os.path.dirname(__file__)}/data/OND/image_classification/image_classification.json",
        }
        config_name = CONFIG_NAME
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield config_folder, config_name


@pytest.fixture(scope="function")
def get_ar_interface_params():
    """Fixture to create a temporary directory and add a configuration.json in it for activity recognition."""
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "url": URL,
            "data_dir": f"{os.path.dirname(__file__)}/data",
            "gt_dir": f"{os.path.dirname(__file__)}/data/OND/activity_recognition",
            "gt_config": f"{os.path.dirname(__file__)}/data/OND/activity_recognition/activity_recognition.json",
        }
        config_name = CONFIG_NAME
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield config_folder, config_name


@pytest.fixture(scope="function")
def discoverable_plugins():
    """
    Fixture to replicate plugin discovery from framework.

    TODO: Replace this with a function call from framework
    """
    discovered_plugins = {}
    for entry_point in iter_entry_points("tinker"):
        try:
            ep = entry_point.load()
            discovered_plugins[entry_point.name] = ep
        except (DistributionNotFound, ImportError):
            log.exception(f"Plugin {entry_point.name} not found")
    return discovered_plugins
