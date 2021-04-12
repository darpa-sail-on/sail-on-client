"""Tests for Base Class For Feedback."""

from tempfile import TemporaryDirectory
import json
import pytest
import os

from sail_on_client.feedback.feedback import Feedback
from sail_on_client.protocol.parinterface import ParInterface


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
    test_id = "OND.54011215.0000.1236"
    # Testing if session was sucessfully initalized
    session_id = par_interface.session_request(
        [test_id], f"{protocol_name}", "image_classification", "0.1.1", list(hints), 0.5
    )
    return session_id, test_id


@pytest.fixture(scope="function")
def ond_config():
    """Fixture to create a temporal directory and create a json file in it."""
    test_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(test_dir, "mock_results", "image_classification")
    with TemporaryDirectory() as config_folder:
        dummy_config = {
            "domain": "image_classification",
            "test_ids": ["OND.54011215.0000.1236"],
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
    ic_feedback = Feedback(
        10, 10, 10, par_interface, session_id, test_id, protocol_constant
    )
    assert ic_feedback.get_budget() == 10
