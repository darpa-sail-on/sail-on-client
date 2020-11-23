"""Tests for Image Classification Feedback."""

from tempfile import TemporaryDirectory
import json
import pytest
import os

from sail_on_client.feedback.image_classification_feedback import (
    ImageClassificationFeedback,
)
from sail_on_client.protocol.parinterface import ParInterface


def _initialize_session(par_interface, protocol_name, hints=()):
    """
    Private function to initialize session.

    Args:
        par_interface (ParInterface): An instance of ParInterface
        protocol_name (str): Name of the protocol
        hints (list[str]): Hints used in session request

    Return:
        session id, test_ids
    """
    test_id_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        f"{protocol_name}",
        "image_classification",
        "test_ids.csv",
    )
    test_ids = list(map(str.strip, open(test_id_path, "r").readlines()))
    # Testing if session was sucessfully initalized
    session_id = par_interface.session_request(
        test_ids, f"{protocol_name}", "image_classification", "0.1.1", list(hints)
    )
    return session_id, test_ids


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


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_initialize(
    server_setup, get_interface_params, feedback_mapping, protocol_name
):
    """
    Test feedback initialization.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id, test_ids = _initialize_session(par_interface, protocol_name)
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )
    par_interface.post_results(result_files, f"{protocol_name}.1.1.1234", 0, session_id)
    ImageClassificationFeedback(
        2, 2, 2, par_interface, session_id, test_ids[0], protocol_constant
    )


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_get_feedback(
    server_setup, get_interface_params, feedback_mapping, protocol_name
):
    """
    Test get feedback.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id, test_ids = _initialize_session(par_interface, protocol_name)
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )
    par_interface.post_results(result_files, f"{protocol_name}.1.1.1234", 0, session_id)
    ic_feedback = ImageClassificationFeedback(
        2, 2, 2, par_interface, session_id, test_ids[0], protocol_constant
    )
    df_feedback = ic_feedback.get_feedback(
        0, [0, 1], ["n01484850_18013.JPEG", "n01484850_24624.JPEG"]
    )
    expected_list = [["n01484850_18013.JPEG", 1], ["n01484850_24624.JPEG", 2]]
    assert df_feedback.values.tolist() == expected_list


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_deposit_income(
    server_setup, get_interface_params, feedback_mapping, protocol_name
):
    """
    Test deposit income.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id, test_ids = _initialize_session(par_interface, protocol_name)
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )
    par_interface.post_results(result_files, f"{protocol_name}.1.1.1234", 0, session_id)
    ic_feedback = ImageClassificationFeedback(
        2, 2, 2, par_interface, session_id, test_ids[0], protocol_constant
    )
    ic_feedback.deposit_income()


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_get_budget(
    server_setup, get_interface_params, feedback_mapping, protocol_name
):
    """
    Test get budget.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id, test_ids = _initialize_session(par_interface, protocol_name)
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )
    par_interface.post_results(result_files, f"{protocol_name}.1.1.1234", 0, session_id)
    ic_feedback = ImageClassificationFeedback(
        2, 2, 2, par_interface, session_id, test_ids[0], protocol_constant
    )
    assert ic_feedback.get_budget() == 2
