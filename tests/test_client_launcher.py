"""Tests for client launcher."""

from hydra import compose, initialize
from tempfile import TemporaryDirectory
import os
import pytest
from sail_on_client.client_launcher import client_launcher


@pytest.fixture(scope="function")
def temp_save_dir():
    """Fixture to provide parameters for the launcher."""
    with TemporaryDirectory() as save_dir:
        yield save_dir


def test_client_launcher(temp_save_dir):
    """
    Test for client launcher.

    Args:
        temp_save_dir: Path to temporary_directory
    """
    test_root = os.path.dirname(__file__)
    config_module = "sail_on_client.configs"
    initialize(config_path=None)
    overrides = [
        f"test_root={test_root}",
        f"protocol.smqtk.config.save_dir={temp_save_dir}",
        f"hydra.searchpath=[pkg://{config_module}]",
    ]
    cfg = compose(config_name="test_config", overrides=overrides)
    client_launcher(cfg)
