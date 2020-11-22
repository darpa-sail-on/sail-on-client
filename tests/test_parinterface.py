"""Tests for PAR Interface."""

import os
import pytest


from sail_on_client.errors import ProtocolError


def _initialize_session(par_interface, protocol_name, hints=()):
    """
    Private function to initialize session.

    Args:
        par_interface (ParInterface): An instance of ParInterface
        protocol_name (str): Name of the protocol
        hints (list[str]): Hints used in session request

    Return:
        session id
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


def test_initialize(server_setup, get_interface_params):
    """
    Test par interface initialization.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    assert par_interface.api_url == server_setup[0]


def test_test_ids_request(server_setup, get_interface_params):
    """
    Test request for test ids.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)

    assumptions_path = os.path.join(os.path.dirname(__file__), "assumptions.json")
    filename = par_interface.test_ids_request(
        "OND", "image_classification", "5678", assumptions_path
    )
    expected = os.path.join(config_directory, "OND.image_classification.5678.csv")
    assert os.stat(expected).st_size > 5
    assert expected == filename


def test_session_request(server_setup, get_interface_params):
    """
    Test session request.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    test_id_path = os.path.join(
        os.path.dirname(__file__), "data", "OND", "image_classification", "test_ids.csv"
    )
    test_ids = list(map(str.strip, open(test_id_path, "r").readlines()))
    # Testing if session was sucessfully initalized
    par_interface.session_request(test_ids, "OND", "image_classification", "0.1.1", [])
    # Testing with hints
    par_interface.session_request(
        test_ids, "OND", "image_classification", "0.1.1", ["red_light"]
    )


def test_dataset_request(server_setup, get_interface_params):
    """
    Tests for dataset request.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id = _initialize_session(par_interface, "OND")
    # Test correct dataset request
    filename = par_interface.dataset_request("OND.1.1.1234", 0, session_id)
    expected = os.path.join(config_directory, f"{session_id}.OND.1.1.1234.0.csv")
    assert expected == filename
    expected_image_ids = _read_image_ids(expected)
    assert expected_image_ids == ["n01484850_18013.JPEG", "n01484850_24624.JPEG"]
    # Test incorrect dataset request
    with pytest.raises(ProtocolError):
        par_interface.dataset_request(f"{session_id}", "OND.1.1.1234", 3)


@pytest.mark.parametrize(
    "protocol_constant", ["detection", "classification", "characterization"]
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_post_results(
    server_setup, get_interface_params, protocol_constant, protocol_name
):
    """
    Tests for post results.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        protocol_constant (str): Constants used by the server to identifying results
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)
    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id = _initialize_session(par_interface, protocol_name)
    result_files = {
        protocol_constant: os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )
    }
    par_interface.post_results(result_files, f"{protocol_name}.1.1.1234", 0, session_id)


@pytest.mark.parametrize(
    "feedback_mapping",
    (
        ("classification", ("detection", "classification")),
        ("psuedo_labels_classification", ("detection", "classification")),
    ),
)
@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_feedback_request(
    server_setup, get_interface_params, feedback_mapping, protocol_name
):
    """
    Tests for feedback request.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface
        feedback_mapping (dict): Dict with mapping for feedback
        protocol_name (str): Name of the protocol ( options: OND and CONDDA)

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id = _initialize_session(par_interface, protocol_name)
    # Post results before posting
    result_files = {}
    protocol_constant = feedback_mapping[0]
    required_files = feedback_mapping[1]
    for required_file in required_files:
        result_files[required_file] = os.path.join(
            os.path.dirname(__file__), f"test_results_{protocol_name}.1.1.1234.csv"
        )

    par_interface.post_results(result_files, f"{protocol_name}.1.1.1234", 0, session_id)
    # Get feedback for detection
    response = par_interface.get_feedback_request(
        ["n01484850_18013.JPEG", "n01484850_24624.JPEG"],
        protocol_constant,
        f"{protocol_name}.1.1.1234",
        0,
        session_id,
    )
    expected = os.path.join(
        config_directory,
        f"{session_id}.{protocol_name}.1.1.1234.0_{protocol_constant}.csv",
    )
    assert expected == response


def test_evaluate(server_setup, get_interface_params):
    """
    Test evaluate with rounds.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id = _initialize_session(par_interface, "OND")
    response = par_interface.evaluate("OND.1.1.1234", 0, session_id)
    expected = os.path.join(
        config_directory, f"{session_id}.OND.1.1.1234.0_evaluation.csv"
    )
    assert expected == response


def test_terminate_session(server_setup, get_interface_params):
    """
    Test terminate session request.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id = _initialize_session(par_interface, "OND")
    par_interface.terminate_session(session_id)


def test_get_metadata(server_setup, get_interface_params):
    """
    Test get metadata.

    Args:
        server_setup (tuple): Tuple containing url and result directory
        get_interface_params (tuple): Tuple to configure par interface

    Return:
        None
    """
    from sail_on_client.protocol.parinterface import ParInterface

    url, result_dir = server_setup
    config_directory, config_name = get_interface_params
    par_interface = ParInterface(config_name, config_directory)
    session_id = _initialize_session(par_interface, "OND")
    metadata = par_interface.get_test_metadata(session_id, "OND.1.1.1234")

    assert "OND" == metadata["protocol"]
    assert 3 == metadata["known_classes"]

    session_id = _initialize_session(par_interface, "OND", ["red_light"])
    metadata = par_interface.get_test_metadata(session_id, "OND.1.1.1234")
    assert "n01484850_4515.JPEG" == metadata["red_light"]
