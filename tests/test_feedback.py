"""Tests for Base Class For Feedback."""

import pytest
import os

from sail_on_client.feedback.feedback import Feedback
from sail_on_client.feedback.image_classification_feedback import (
    ImageClassificationFeedback,
)
from sail_on_client.feedback.document_transcription_feedback import (
    DocumentTranscriptionFeedback,
)
from sail_on_client.feedback.activity_recognition_feedback import (
    ActivityRecognitionFeedback,
)
from sail_on_client.feedback import create_feedback_instance
from sail_on_client.harness.local_harness import LocalHarness
from sail_on_client.harness.par_harness import ParHarness


feedback_image_ids = [
    "known_classes/images/val/00233/00068.JPEG",
    "known_classes/images/val/00072/00048.JPEG",
    "known_classes/images/val/00330/00013.JPEG",
    "known_classes/images/val/00099/00032.JPEG",
    "known_classes/images/val/00228/00091.JPEG",
    "known_classes/images/val/00200/00062.JPEG",
    "known_classes/images/val/00080/00085.JPEG",
    "known_classes/images/val/00277/00062.JPEG",
    "known_classes/images/val/00004/00073.JPEG",
    "known_classes/images/val/00365/00047.JPEG",
]

feedback_labels = [233, 72, 330, 99, 228, 200, 80, 277, 4, 365]


def _initialize_session(par_harness, protocol_name, hints=()):
    """
    Private function to initialize session.

    Args:
        par_harness (ParHarness): An instance of ParHarness
        protocol_name (str): Name of the protocol
        hints (list[str]): Hints used in session request

    Return:
        session id, test_ids
    """
    test_id = "OND.54011215.0000.1236"
    # Testing if session was sucessfully initalized
    session_id = par_harness.session_request(
        [test_id], f"{protocol_name}", "image_classification", "0.1.1", list(hints), 0.5
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
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    Feedback(10, 10, 10, par_interface, session_id, test_id, protocol_constant)


@pytest.mark.parametrize(
    "feedback_mapping", (("classification", ("detection", "classification")),)
)
@pytest.mark.parametrize("protocol_name", ["OND"])
def test_get_labeled_feedback(
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
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    feedback = Feedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    df_labelled = feedback.get_labeled_feedback(0, list(range(10)), feedback_image_ids)
    assert all(df_labelled.id == feedback_image_ids)
    assert all(df_labelled.labels == feedback_labels)


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
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    feedback = Feedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    df_score = feedback.get_feedback(0, list(range(10)), feedback_image_ids)
    assert df_score[1][0] == 0.59921875


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
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            result_folder, f"{test_id}_{required_file}.csv"
        )
    par_interface.post_results(result_files, f"{test_id}", 0, session_id)
    feedback = Feedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    feedback.get_feedback(0, list(range(10)), feedback_image_ids)


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("score", ("detection", "classification")),
    ),
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
    feedback = Feedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    feedback.deposit_income()
    assert feedback.budget == 10


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("score", ("detection", "classification")),
    ),
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
    ic_feedback = Feedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    assert ic_feedback.get_budget() == 10


@pytest.mark.parametrize(
    "domain,test_id,expected",
    [
        ("image_classification", "OND.54011215.0000.1236", ImageClassificationFeedback),
        ("transcripts", "OND.0.90001.8714062", DocumentTranscriptionFeedback),
        ("activity_recognition", "OND.10.90001.2100554", ActivityRecognitionFeedback),
    ],
)
def test_create_feedback_instance(domain, test_id, get_local_harness_params, expected):
    """
    Test for creating metric instance.

    Args:
        protocol: Name of the protocol
        domain: Name of the domain
        gt_dict: Parameters for the class created by metric
        expected: Expected Output Class

    Returns:
        None
    """
    data_dir, gt_dir, gt_config = get_local_harness_params
    local_harness = LocalHarness(data_dir, gt_dir, gt_config)
    protocol_name = "OND"
    session_id = local_harness.session_request(
        [test_id], protocol_name, domain, "0.1.1", (), 0.5
    )
    feedback_dict = {
        "first_budget": 10,
        "income_per_batch": 10,
        "maximum_budget": 10,
        "interface": local_harness,
        "session_id": session_id,
        "test_id": test_id,
        "feedback_type": "classification",
    }
    feedback_obj = create_feedback_instance(domain, feedback_dict)
    assert isinstance(feedback_obj, expected)
