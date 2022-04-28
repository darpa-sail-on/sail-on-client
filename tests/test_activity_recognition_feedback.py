"""Tests for Activity Recognition Feedback."""

import pytest
import os
import numpy as np

from sail_on_client.feedback.activity_recognition_feedback import (
    ActivityRecognitionFeedback,
)
from sail_on_client.harness.par_harness import ParHarness

FEEDBACK_BUDGET = 5

feedback_image_ids = [
    "91e68de4-92d1-4e4e-b6d1-8d01a8fe2cf8.mp4",
    "7035526b-4d00-4ffa-9177-2c02dd13834b.mp4",
    "43c3c705-a57b-4e2b-8cd0-91fc15e34449.mp4",
    "7fcebe2f-e133-4ad1-ae27-eb53d0e0b97e.mp4",
    "42cb4d19-0186-4168-a169-b753c57aa8d9.avi",
]


feedback_labels = [22,22,22,22,22]


def _initialize_session(par_interface, protocol_name, hints=()):
    """
    Private function to initialize session.

    Args:
        par_interface (ParHarness): An instance of ParHarness
        protocol_name (str): Name of the protocol
        hints (list[str]): Hints used in session request

    Return:
        session id, test_ids
    """
    test_id = "OND.0.10100.6438158"
    # Testing if session was sucessfully initalized
    session_id = par_interface.session_request(
        [test_id], f"{protocol_name}", "activity_recognition", "0.1.1", list(hints), 0.5
    )
    return session_id, test_id


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_initialize(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test feedback initialization.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    protocol_constant = feedback_mapping[0]
    ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )


@pytest.mark.parametrize(
    "feedback_mapping", (("labels", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_labelled_feedback(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test get feedback.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder,
            f"{test_id}_PreComputed{protocol_name}Agent_{required_file}.csv",
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    ar_feedback = ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )
    df_labelled = ar_feedback.get_feedback(
        0, list(range(FEEDBACK_BUDGET)), feedback_image_ids
    )
    assert all(df_labelled.id == feedback_image_ids)
    assert (
        df_labelled["labels"].tolist()
        == feedback_labels
    )


@pytest.mark.parametrize(
    "feedback_mapping", (("detection", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_detection_feedback(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test get feedback.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder,
            f"{test_id}_PreComputed{protocol_name}Agent_{required_file}.csv",
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    ar_feedback = ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )
    df_labelled = ar_feedback.get_feedback(
        0, list(range(FEEDBACK_BUDGET)), feedback_image_ids
    )
    assert all(df_labelled.id == feedback_image_ids)


@pytest.mark.parametrize(
    "feedback_mapping",
    (("detection_and_classification", ("detection", "classification")),),
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_detection_and_classification_feedback(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test get feedback.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder,
            f"{test_id}_PreComputed{protocol_name}Agent_{required_file}.csv",
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    ar_feedback = ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )
    df_labelled = ar_feedback.get_feedback(
        0, list(range(FEEDBACK_BUDGET)), feedback_image_ids
    )
    assert all(df_labelled.id == feedback_image_ids)


@pytest.mark.parametrize(
    "feedback_mapping", (("score", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_score_feedback(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test get feedback.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder,
            f"{test_id}_PreComputed{protocol_name}Agent_{required_file}.csv",
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    feedback = ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )
    df_score = feedback.get_feedback(
        0, list(range(FEEDBACK_BUDGET)), feedback_image_ids
    )
    assert np.isclose(df_score[1][0], 0.0, atol=1e-05)


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("score", ("detection", "classification")),
    ),
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_feedback(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test get feedback.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    result_files = {}
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder,
            f"{test_id}_PreComputed{protocol_name}Agent_{required_file}.csv",
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    ar_feedback = ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )
    ar_feedback.get_feedback(0, list(range(FEEDBACK_BUDGET)), feedback_image_ids)


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_deposit_income(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test deposit income.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    protocol_constant = feedback_mapping[0]
    ar_feedback = ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )
    ar_feedback.deposit_income()
    assert ar_feedback.budget == FEEDBACK_BUDGET


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_budget(
    server_setup, get_par_harness_params, feedback_mapping, protocol_name
):
    """
    Test get budget.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_par_harness_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    url, save_directory = get_par_harness_params
    par_interface = ParHarness(url, save_directory)
    session_id, test_id = _initialize_session(par_interface, protocol_name)
    protocol_constant = feedback_mapping[0]
    ar_feedback = ActivityRecognitionFeedback(
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        FEEDBACK_BUDGET,
        par_interface,
        session_id,
        test_id,
        protocol_constant,
    )
    assert ar_feedback.get_budget() == FEEDBACK_BUDGET
