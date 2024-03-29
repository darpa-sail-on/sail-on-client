"""Helpers for Tests."""

import pytest
import os
import time
import shutil
import ubelt as ub
import multiprocessing
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from sail_on.api import server
from sail_on_client.harness.file_provider import FileProvider
from sail_on_client.agent.pre_computed_detector import PreComputedONDAgent
from sail_on_client.agent.pre_computed_detector import PreComputedCONDDAAgent
from sail_on_client.harness.local_harness import LocalHarness

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
def get_local_harness_params():
    """Fixture to provide local harness parameters."""
    with TemporaryDirectory() as temp_dir:
        data_dir = f"{os.path.dirname(__file__)}/data"
        gt_dir = f"{data_dir}/OND/image_classification"
        gt_config = f"{data_dir}/OND/image_classification/image_classification.json"
        yield data_dir, temp_dir, gt_dir, gt_config


@pytest.fixture(scope="function")
def get_par_harness_params():
    """Fixture to provide par harness parameters."""
    with TemporaryDirectory() as temp_dir:
        yield URL, temp_dir


@pytest.fixture(scope="function")
def get_ar_local_harness_params():
    """Fixture to create parameters for local harness in activity recognition."""
    with TemporaryDirectory() as temp_dir:
        data_dir = f"{os.path.dirname(__file__)}/data"
        gt_dir = f"{data_dir}/OND/activity_recognition"
        gt_config = f"{data_dir}/OND/activity_recognition/activity_recognition.json"
        yield data_dir, temp_dir, gt_dir, gt_config


@pytest.fixture(scope="function")
def get_dt_local_harness_params():
    """Fixture to create parameters for local harness in document transcription."""
    with TemporaryDirectory() as temp_dir:
        data_dir = f"{os.path.dirname(__file__)}/data"
        gt_dir = f"{data_dir}/OND/transcripts"
        gt_config = f"{data_dir}/OND/transcripts/transcripts.json"
        yield data_dir, temp_dir, gt_dir, gt_config


@pytest.fixture(scope="function")
def ond_harness_instance():
    """Fixture for creating an instance of harness for OND."""
    with TemporaryDirectory() as temp_dir:
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, "data")
        gt_dir = os.path.join(data_dir, "OND", "activity_recognition")
        gt_config = os.path.join(gt_dir, "activity_recognition.json")
        local_interface = LocalHarness(data_dir, temp_dir, gt_dir, gt_config)
        yield local_interface


@pytest.fixture(scope="function")
def ond_algorithm_instance():
    """Fixture for creating an agent for OND."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    return PreComputedONDAgent("PreComputedONDAgent", cache_dir, False, 32)


@pytest.fixture(scope="function")
def condda_harness_instance():
    """Fixture for creating an instance of harness for OND."""
    with TemporaryDirectory() as temp_dir:
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, "data")
        gt_dir = os.path.join(data_dir, "CONDDA", "activity_recognition")
        gt_config = os.path.join(
            data_dir, "OND", "activity_recognition", "activity_recognition.json"
        )
        local_interface = LocalHarness(data_dir, temp_dir, gt_dir, gt_config)
        yield local_interface


@pytest.fixture(scope="function")
def condda_algorithm_instance():
    """Fixture for creating an agent for OND."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
    return PreComputedCONDDAAgent("PreComputedCONDDAAgent", cache_dir, False, 32)
