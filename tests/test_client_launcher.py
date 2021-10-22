"""Tests for client launcher."""

from hydra import compose, initialize_config_module, initialize_config_dir
from tempfile import TemporaryDirectory
import os
import pytest
from smqtk_core import Pluggable
from sail_on_client.protocol.visual_protocol import VisualProtocol
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
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config", "test")
    test_root = os.path.dirname(__file__)
    config_module = "sail_on_client.configs"
    initialize_config_dir(config_dir=config_dir)
    overrides = [f"test_root={test_root}",
                 f"protocol.smqtk.config.save_dir={temp_save_dir}",
                 f"hydra.searchpath=[pkg://{config_module}]"]
    cfg = compose(config_name="test_config", overrides=overrides)
    client_launcher(cfg)
