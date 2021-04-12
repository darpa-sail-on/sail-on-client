"""Tests for Activity Recognition Feedback."""

from tempfile import TemporaryDirectory
import json
import pytest
import os

from sail_on_client.feedback.activity_recognition_feedback import (
    ActivityRecognitionFeedback,
)
from sail_on_client.protocol.parinterface import ParInterface

feedback_image_ids = [
    "3985211f-d6dd-4510-81d7-1825469008f3.avi",
    "ed60328c-1cfb-4082-8586-f9b5902d79eb.avi",
    "2c2c4800-6ddf-4bf3-be01-a83635198720.avi",
    "2df4df62-fa02-4638-a054-e6a5fe19edee.avi",
    "1a50f7c1-c0c6-46d6-a3ab-05959f9f5545.avi",
    "3e800115-770f-4067-8543-af77c8661133.avi",
    "31d4b650-3f4f-41b7-9f7f-93433d470bee.avi",
    "c0784443-e0e7-4b76-817c-fc6e1529f467.avi",
    "de9af377-2e0b-42da-9b91-de79e13537d9.avi",
    "f97a7f62-7db3-46c9-b827-8f2676760a04.avi"
]


feedback_labels = [
    74,
    34,
    10,
    10,
    21,
    21,
    21,
    10,
    74,
    21
]


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
    test_id = "OND.10.90001.2100554"
    # Testing if session was sucessfully initalized
    session_id = par_interface.session_request(
        [test_id], f"{protocol_name}", "activity_recognition", "0.1.1", list(hints), 0.5
    )
    return session_id, test_id


@pytest.fixture(scope="function")
def ond_config():
    """Fixture to create a temporal directory and create a json file in it."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "activity_recognition")
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
        }
        config_name = "test_ond_config.json"
        json.dump(dummy_config, open(os.path.join(config_folder, config_name), "w"))
        yield os.path.join(config_folder, config_name)


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
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
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    protocol_constant = feedback_mapping[0]
    ActivityRecognitionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_labelled_feedback(
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
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_PreComputedDetector_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    ar_feedback = ActivityRecognitionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    df_labelled = ar_feedback.get_feedback(
        0, list(range(10)), feedback_image_ids
    )
    assert all(df_labelled.id == feedback_image_ids)
    assert all(df_labelled.labels == feedback_labels)


@pytest.mark.parametrize(
    "feedback_mapping", (("score", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_score_feedback(
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
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_PreComputedDetector_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    feedback = ActivityRecognitionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    df_score = feedback.get_feedback(
        0, list(range(10)), feedback_image_ids
    )
    assert df_score[1][0] == 0.1987951807228915


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),
                         ("score", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
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
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_PreComputedDetector_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    ar_feedback = ActivityRecognitionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    ar_feedback.get_feedback(
        0, list(range(10)), feedback_image_ids
    )


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
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
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    protocol_constant = feedback_mapping[0]
    ar_feedback = ActivityRecognitionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    ar_feedback.deposit_income()
    assert ar_feedback.budget == 10


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
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
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    protocol_constant = feedback_mapping[0]
    ar_feedback = ActivityRecognitionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    assert ar_feedback.get_budget() == 10
