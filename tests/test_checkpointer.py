"""Tests for Checkpointer."""

from sail_on_client.checkpointer import Checkpointer
from sail_on_client.mock import MockAdapterWithCheckpoint
from tempfile import TemporaryDirectory
from random import randint
import torch
import os
import ubelt as ub
import pytest
import pickle as pkl


@pytest.fixture(scope="function")
def checkpoint_save_config():
    """Fixture to create a config for saving attributes of a detector."""
    toolset = {
        "test_id": "Dummy_test",
        "saved_attributes": {
            "FeatureExtraction": [
                "dummy_dict",
                "dummy_list",
                "dummy_tuple",
                "dummy_tensor",
                "dummy_val",
            ],
        },
        "save_attributes": True,
        "attributes": {},
        "save_elementwise": True,
    }
    return toolset


@pytest.fixture(scope="function")
def checkpoint_restore_config(request):
    """Fixture to create a config for restoring attributes of a detector."""
    save_dir = request.param
    # Check if save dir is a directory
    if os.path.isdir(save_dir):
        ub.ensuredir(save_dir)
        dataset_file = os.path.join(save_dir, "dummy_dataset.txt")
    else:
        dir_name = os.path.dirname(save_dir)
        ub.ensuredir(dir_name)
        dataset_file = os.path.join(dir_name, "dummy_dataset.txt")

    # Create a dataset with 1 image to indicate 1 image in a round
    with open(dataset_file, "w") as f:
        f.writelines(["dummy_key\n"])

    toolset = {
        "test_id": "Dummy_test",
        "save_dir": save_dir,
        "dataset": dataset_file,
        "round_id": 0,
        "saved_attributes": {
            "FeatureExtraction": [
                "dummy_dict",
                "dummy_list",
                "dummy_tuple",
                "dummy_tensor",
                "dummy_val",
            ],
        },
        "use_saved_attributes": True,
        "save_elementwise": True,
    }
    return toolset


@pytest.fixture(scope="function")
def dummy_attributes():
    """Fixture for generating dummy attributes."""
    toolset = {
        "dummy_dict": {"dummy_key": "Dummy_val"},
        "dummy_list": [randint(1, 100)],
        "dummy_tuple": (randint(1, 100)),
        "dummy_tensor": torch.rand(1, 10),
        "dummy_val": randint(1, 100),
    }
    return toolset


def test_initialize():
    """
    Test checkpointer initialization.

    Return:
        None
    """
    checkpointer = Checkpointer({"test_id": "Dummy_test"})
    assert checkpointer.toolset["test_id"] == "Dummy_test"


def test_save_attribute(checkpoint_save_config, dummy_attributes):
    """
    Test attributes saved by checkpointer.

    Args:
        checkpoint_save_config (dict): Dictionary with the config to save attributes
        dummy_attributes (dict): Dictionary with value for the attributes
    Return:
        None
    """
    mock_detector = MockAdapterWithCheckpoint(checkpoint_save_config)
    mock_detector.execute(dummy_attributes, "FeatureExtraction")
    mock_detector.save_attributes("FeatureExtraction")
    attribute_dict = mock_detector.toolset["attributes"]
    test_id = checkpoint_save_config["test_id"]
    for attribute_name in attribute_dict:
        attribute_val = attribute_dict[attribute_name][test_id]
        dummy_val = dummy_attributes[attribute_name]
        if isinstance(attribute_val, torch.Tensor):
            assert torch.all(torch.eq(attribute_val, dummy_val))
        else:
            assert attribute_val == dummy_val


@pytest.mark.parametrize(
    "checkpoint_restore_config",
    [
        TemporaryDirectory().name,
        os.path.join(TemporaryDirectory().name, "attributes.pkl"),
    ],
    indirect=True,
)
def test_restore_attribute(
    checkpoint_restore_config, checkpoint_save_config, dummy_attributes
):
    """
    Test attributes restored by checkpointer.

    Args:
        checkpoint_save_config (dict): Dictionary with the config to save attributes
        checkpoint_restore_config (dict): Dictionary with the config to save attributes
        dummy_attributes (dict): Dictionary with value for the attributes
    Return:
        None
    """
    mock_detector = MockAdapterWithCheckpoint(checkpoint_save_config)
    mock_detector.execute(dummy_attributes, "FeatureExtraction")
    mock_detector.save_attributes("FeatureExtraction")
    attribute_dict = mock_detector.toolset["attributes"]
    save_dir = checkpoint_restore_config["save_dir"]
    if os.path.isdir(save_dir):
        save_dir = os.path.join(save_dir, "Dummy_test_attribute.pkl")
    pkl.dump(attribute_dict, open(save_dir, "wb"))
    restored_mock_detector = MockAdapterWithCheckpoint(checkpoint_restore_config)
    restored_mock_detector.restore_attributes("FeatureExtraction")
    assert mock_detector == restored_mock_detector
