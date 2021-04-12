"""Tests for Image Classification Feedback."""

from tempfile import TemporaryDirectory
import json
import pytest
import os

from sail_on_client.feedback.document_transcription_feedback import (
    DocumentTranscriptionFeedback,
)
from sail_on_client.protocol.parinterface import ParInterface


feedback_image_ids = [
    "61b1747b-84f0-4ef7-b11f-691d842c524b.png",
    "659ec8f9-aefd-4645-9d1d-e3e0a2356e2b.png",
    "df523435-bb28-47a0-a42a-17b9e7e5f11d.png",
    "13efc410-8fab-41ed-a840-1beb9648613c.png",
    "e5290f9e-a313-411e-9801-691db75c33ae.png",
    "17ea4fe5-db52-4e50-9765-4d31b0e3eae2.png",
    "302b940a-c985-485c-a50f-df615fca38e7.png",
    "e0e7ae2c-91e5-4ccf-bb03-eb469f38c316.png",
    "11899acf-a701-4180-bc20-00cc0ac47482.png",
    "823ecc48-70f0-4d19-82d7-2a6d1e240d44.png",
]


feedback_labels = [44, 4, 9, 45, 0, 9, 0, 35, 35, 15]


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
    test_id = "OND.0.90001.8714062"
    # Testing if session was sucessfully initalized
    session_id = par_interface.session_request(
        [test_id], f"{protocol_name}", "transcripts", "0.1.1", list(hints), 0.5
    )
    return session_id, test_id


@pytest.fixture(scope="function")
def ond_config():
    """Fixture to create a temporal directory and create a json file in it."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "transcripts")
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "domain": "transcripts",
            "test_ids": ["OND.0.90001.8714062"],
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
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    DocumentTranscriptionFeedback(
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
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    dt_feedback = DocumentTranscriptionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    df_labelled = dt_feedback.get_feedback(0, list(range(10)), feedback_image_ids)
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
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    dt_feedback = DocumentTranscriptionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    df_score = dt_feedback.get_feedback(0, list(range(10)), feedback_image_ids)
    assert df_score[1][0] == 0.76953125


@pytest.mark.parametrize(
    "feedback_mapping",
    (("transcription", ("detection", "classification", "transcription")),),
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_lavenshtein_feedback(
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
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    dt_feedback = DocumentTranscriptionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    df_score = dt_feedback.get_feedback(0, list(range(10)), feedback_image_ids)
    assert df_score[1][0] == 8


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("score", ("detection", "classification")),
        ("transcription", ("detection", "classification", "transcription")),
    ),
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
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    dt_feedback = DocumentTranscriptionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    dt_feedback.get_feedback(0, list(range(10)), feedback_image_ids)


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("score", ("detection", "classification")),
    ),
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
    dt_feedback = DocumentTranscriptionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    dt_feedback.deposit_income()
    assert dt_feedback.budget == 10


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("score", ("detection", "classification")),
    ),
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
    dt_feedback = DocumentTranscriptionFeedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    assert dt_feedback.get_budget() == 10
