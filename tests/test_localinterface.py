"""Tests for PAR Interface."""

import os
import pytest


TEST_ID_NAME = "test_ids.csv"


def _initialize_session(
    local_interface, protocol_name, domain="image_classification", hints=()
):
    """
    Private function to initialize session.

    Args:
        local_interface (LocalInterface): An instance of LocalInterface
        protocol_name (str): Name of the protocol
        domain (str): Name of the domain
        hints (list[str]): Hints used in session request

    Return:
        session id
    """
    test_id_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        f"{protocol_name}",
        f"{domain}",
        TEST_ID_NAME,
    )
    test_ids = list(map(str.strip, open(test_id_path, "r").readlines()))
    # Testing if session was sucessfully initalized
    session_id = local_interface.session_request(
        test_ids, f"{protocol_name}", f"{domain}", "0.1.1", list(hints), 0.5
    )
    return session_id


def _read_image_ids(image_ids_path):
    """
    Private function to read image ids from a csv file.

    Args:
        image_ids_path (str): Path to a file containing image ids

    Return:
        list of image ids
    """
    return list(map(str.strip, open(image_ids_path, "r").readlines()))


def test_initialize(get_interface_params):
    """
    Test local interface initialization.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    LocalInterface(config_name, config_directory)


def test_test_ids_request(get_interface_params):
    """
    Test request for test ids.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    data_dir = f"{os.path.dirname(__file__)}/data"
    local_interface = LocalInterface(config_name, config_directory)

    assumptions_path = os.path.join(os.path.dirname(__file__), "assumptions.json")
    filename = local_interface.test_ids_request(
        "OND", "image_classification", "5678", assumptions_path
    )
    expected = os.path.join(data_dir, "OND", "image_classification", TEST_ID_NAME)
    assert os.stat(expected).st_size > 5
    assert expected == filename


def test_session_request(get_interface_params):
    """
    Test session request.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    data_dir = f"{os.path.dirname(__file__)}/data"
    local_interface = LocalInterface(config_name, config_directory)
    test_id_path = os.path.join(data_dir, "OND", "image_classification", TEST_ID_NAME)
    test_ids = list(map(str.strip, open(test_id_path, "r").readlines()))
    # Testing if session was sucessfully initalized
    local_interface.session_request(
        test_ids, "OND", "image_classification", "0.1.1", [], 0.5
    )
    # Testing with hints
    local_interface.session_request(
        test_ids, "OND", "image_classification", "0.1.1", ["red_light"], 0.5
    )


def test_dataset_request(get_interface_params):
    """
    Tests for dataset request.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND")
    # Test correct dataset request
    filename = local_interface.dataset_request("OND.1.1.1234", 0, session_id)
    expected = os.path.join(
        local_interface.result_directory, f"{session_id}.OND.1.1.1234.0.csv"
    )
    assert expected == filename
    expected_image_ids = _read_image_ids(expected)
    assert expected_image_ids == ["n01484850_18013.JPEG", "n01484850_24624.JPEG"]


@pytest.mark.parametrize(
    "protocol_constant", ["detection", "classification", "characterization"]
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_post_results(get_interface_params, protocol_constant, protocol_name):
    """
    Tests for post results.

    Args:
        get_interface_params (tuple): Tuple to configure local interface
        protocol_constant (str): Constants used by the server to identifying results
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)
    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, protocol_name)
    result_files = {
        protocol_constant: os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )
    }
    local_interface.post_results(
        result_files, f"{protocol_name}.1.1.1234", 0, session_id
    )


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("score", ("detection", "classification")),
    ),
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_feedback_request(get_interface_params, feedback_mapping, protocol_name):
    """
    Tests for feedback request.

    Args:
        get_interface_params (tuple): Tuple to configure local interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol (options: OND and CONDDA)

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, protocol_name)
    # Post results before posting
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )
    local_interface.post_results(
        result_files, f"{protocol_name}.1.1.1234", 0, session_id
    )
    # Get feedback for detection
    response = local_interface.get_feedback_request(
        ["n01484850_18013.JPEG", "n01484850_24624.JPEG"],
        protocol_constant,
        f"{protocol_name}.1.1.1234",
        0,
        session_id,
    )
    expected = os.path.join(
        local_interface.result_directory,
        "feedback",
        f"{session_id}.{protocol_name}.1.1.1234.0_{protocol_constant}.csv",
    )
    assert expected == response


def test_image_classification_evaluate(get_ic_interface_params):
    """
    Test evaluate with rounds.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_ic_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND", "image_classification")
    baseline_session_id = _initialize_session(
        local_interface, "OND", "image_classification"
    )
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    detection_file_id = os.path.join(
        result_folder, "OND.54011215.0000.1236_PreComputedDetector_detection.csv"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.54011215.0000.1236_PreComputedDetector_classification.csv"
    )
    baseline_classification_file_id = os.path.join(
        result_folder,
        "OND.54011215.0000.1236_BaselinePreComputedDetector_classification.csv",
    )
    results = {
        "detection": detection_file_id,
        "classification": classification_file_id,
    }
    baseline_result = {
        "classification": baseline_classification_file_id,
    }
    local_interface.post_results(results, "OND.54011215.0000.1236", 0, session_id)
    local_interface.post_results(
        baseline_result, "OND.54011215.0000.1236", 0, baseline_session_id
    )
    local_interface.evaluate("OND.54011215.0000.1236", 0, session_id)
    local_interface.evaluate(
        "OND.54011215.0000.1236", 0, session_id, baseline_session_id
    )


def test_activity_recognition_evaluate(get_ar_interface_params):
    """
    Test evaluate for activity recognition.

    Args:
        get_ar_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_ar_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND", "activity_recognition")
    baseline_session_id = _initialize_session(
        local_interface, "OND", "activity_recognition"
    )
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    detection_file_id = os.path.join(
        result_folder, "OND.10.90001.2100554_PreComputedDetector_detection.csv"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.10.90001.2100554_PreComputedDetector_classification.csv"
    )
    characterization_file_id = os.path.join(
        result_folder, "OND.10.90001.2100554_PreComputedDetector_characterization.csv"
    )
    results = {
        "detection": detection_file_id,
        "classification": classification_file_id,
        "characterization": characterization_file_id,
    }
    baseline_classification_file_id = os.path.join(
        result_folder,
        "OND.10.90001.2100554_BaselinePreComputedDetector_classification.csv",
    )
    baseline_result = {
        "classification": baseline_classification_file_id,
    }
    local_interface.post_results(results, "OND.10.90001.2100554", 0, session_id)
    local_interface.post_results(
        baseline_result, "OND.10.90001.2100554", 0, baseline_session_id
    )
    local_interface.evaluate("OND.10.90001.2100554", 0, session_id)
    local_interface.evaluate("OND.10.90001.2100554", 0, session_id, baseline_session_id)


def test_transcripts_evaluate(get_transcripts_interface_params):
    """
    Test evaluate for transcripts.

    Args:
        get_transcripts_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_transcripts_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND", "transcripts")
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    detection_file_id = os.path.join(
        result_folder, "OND.0.90001.8714062_PreComputedDetector_detection.csv"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.0.90001.8714062_PreComputedDetector_classification.csv"
    )
    characterization_file_id = os.path.join(
        result_folder, "OND.0.90001.8714062_PreComputedDetector_characterization.csv"
    )
    results = {
        "detection": detection_file_id,
        "classification": classification_file_id,
        "characterization": characterization_file_id,
    }

    baseline_session_id = _initialize_session(local_interface, "OND", "transcripts")
    local_interface.post_results(results, "OND.0.90001.8714062", 0, session_id)
    local_interface.evaluate("OND.0.90001.8714062", 0, session_id)

    baseline_classification_file_id = os.path.join(
        result_folder,
        "OND.0.90001.8714062_BaselinePreComputedDetector_classification.csv",
    )
    baseline_result = {
        "classification": baseline_classification_file_id,
    }
    local_interface.post_results(
        baseline_result, "OND.0.90001.8714062", 0, baseline_session_id
    )
    local_interface.evaluate("OND.0.90001.8714062", 0, session_id, baseline_session_id)


def test_image_classification_evaluate_roundwise(get_ic_interface_params):
    """
    Test evaluate with rounds.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_ic_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND", "image_classification")
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    detection_file_id = os.path.join(
        result_folder, "OND.54011215.0000.1236_PreComputedDetector_detection.csv"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.54011215.0000.1236_PreComputedDetector_classification.csv"
    )
    results = {
        "detection": detection_file_id,
        "classification": classification_file_id,
    }
    local_interface.post_results(results, "OND.54011215.0000.1236", 0, session_id)
    local_interface.evaluate_round_wise("OND.54011215.0000.1236", 0, session_id)


def test_complete_test(get_interface_params):
    """
    Test complete test request.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND")
    local_interface.complete_test(session_id, "OND.10.90001.2100554")


def test_terminate_session(get_interface_params):
    """
    Test terminate session request.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND")
    local_interface.terminate_session(session_id)


def test_get_metadata(get_interface_params):
    """
    Test get metadata.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    from sail_on_client.protocol.localinterface import LocalInterface

    config_directory, config_name = get_interface_params
    local_interface = LocalInterface(config_name, config_directory)
    session_id = _initialize_session(local_interface, "OND")
    metadata = local_interface.get_test_metadata(session_id, "OND.1.1.1234")

    assert "OND" == metadata["protocol"]
    assert 3 == metadata["known_classes"]

    session_id = _initialize_session(local_interface, "OND", hints=["red_light"])
    metadata = local_interface.get_test_metadata(session_id, "OND.1.1.1234")
    assert "n01484850_4515.JPEG" == metadata["red_light"]
