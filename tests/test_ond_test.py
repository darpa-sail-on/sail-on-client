"""Tests for ONDTest."""

import os
import pickle as pkl
import numpy as np
from tempfile import TemporaryDirectory

from sail_on_client.protocol.ond_test import ONDTest
from sail_on_client.protocol.ond_dataclasses import AlgorithmAttributes

DOMAIN = "activity_recognition"
PROTOCOL = "OND"
FEEDBACK_TYPE = "classification"
ALGORITHM_NAME = "PreComputedDetector"


def create_algorithm_attribute(
    algorithm_name,
    instance,
    is_baseline,
    is_reaction_baseline,
    parameters,
    session_id,
    test_ids,
):
    """
    Create algorithm attributes.

    Args:
        algorithm_name: Name of the algorithm
        instance: An instance of the algorithm
        is_baseline: Flag to specify if the algorithm is baseline
        is_reaction_baseline: Flag to specify if algorithm is reaction baseline
        parameters: Parameters of the algorithm
        session_id: Session id for the algorithm
        test_ids: Tests associated with the algorithm

    Returns:
        An instance of AlgorithmAttributes
    """
    return AlgorithmAttributes(
        algorithm_name,
        0.5,
        instance,
        is_baseline,
        is_reaction_baseline,
        "sail-on-client",
        parameters,
        session_id,
        test_ids,
    )


def create_temp_features(save_dir, test_id):
    """
    Create features for a test id.

    Args:
        test_id: Test identifier

    Yields:
        temporary directory where features are stored
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", PROTOCOL, DOMAIN)
    instance_ids = map(
        lambda x: x.strip(),
        open(os.path.join(data_dir, f"{test_id}.csv"), "r").readlines(),
    )
    pkl_path = os.path.join(save_dir, f"{ALGORITHM_NAME}_features.pkl")
    features = {"features_dict": {}, "logit_dict": {}}
    for instance_id in instance_ids:
        features["features_dict"][instance_id] = np.random.random(1024)
        features["logit_dict"][instance_id] = np.random.random(88)
    with open(pkl_path, "wb") as f:
        pkl.dump(features, f)


def test_initialize(ond_harness_instance, ond_algorithm_instance):
    """
    Test initialization of ONDTest.

    Args:
        ond_harness_instance: Instance of local interface
        ond_algorithm_instance: Instance of PreComputedDetector

    Returns:
        None
    """
    test_ids = ["OND.10.90001.2100554"]
    algorithm_attribute = create_algorithm_attribute(
        ALGORITHM_NAME, ond_algorithm_instance, False, False, {}, "", test_ids
    )
    session_id = ond_harness_instance.session_request(
        test_ids,
        PROTOCOL,
        DOMAIN,
        algorithm_attribute.named_version(),
        [],
        algorithm_attribute.detection_threshold,
    )
    algorithm_attribute.session_id = session_id
    ONDTest(
        algorithm_attribute,
        "",
        DOMAIN,
        FEEDBACK_TYPE,
        "",
        ond_harness_instance,
        "",
        session_id,
        [],
        False,
        False,
    )


def test_call(ond_harness_instance, ond_algorithm_instance):
    """
    Test __call__ of ONDTest.

    Args:
        ond_harness_instance: Instance of local interface
        ond_algorithm_instance: Instance of PreComputedDetector

    Returns:
        None
    """
    test_ids = ["OND.10.90001.2100554"]
    feedback_params = {"first_budget": 4, "income_per_batch": 4, "maximum_budget": 4}
    algorithm_attribute = create_algorithm_attribute(
        ALGORITHM_NAME,
        ond_algorithm_instance,
        False,
        False,
        {"feedback_params": feedback_params},
        "",
        test_ids,
    )
    session_id = ond_harness_instance.session_request(
        test_ids,
        PROTOCOL,
        DOMAIN,
        algorithm_attribute.named_version(),
        [],
        algorithm_attribute.detection_threshold,
    )
    algorithm_attribute.session_id = session_id
    ond_test = ONDTest(
        algorithm_attribute,
        "",
        DOMAIN,
        "classification",
        "",
        ond_harness_instance,
        "",
        session_id,
        [],
        False,
        False,
    )
    ond_test(test_ids[0])


def test_call_with_features(ond_harness_instance, ond_algorithm_instance):
    """
    Test __call__ of ONDTest with features.

    Args:
        ond_harness_instance: Instance of local interface
        ond_algorithm_instance: Instance of PreComputedDetector

    Returns:
        None
    """
    test_ids = ["OND.10.90001.2100554"]
    feedback_params = {"first_budget": 4, "income_per_batch": 4, "maximum_budget": 4}
    algorithm_attribute = create_algorithm_attribute(
        ALGORITHM_NAME,
        ond_algorithm_instance,
        False,
        False,
        {"feedback_params": feedback_params},
        "",
        test_ids,
    )
    session_id = ond_harness_instance.session_request(
        test_ids,
        PROTOCOL,
        DOMAIN,
        algorithm_attribute.named_version(),
        [],
        algorithm_attribute.detection_threshold,
    )
    with TemporaryDirectory() as tempdirectory:
        create_temp_features(tempdirectory, test_ids[0])
        algorithm_attribute.session_id = session_id
        ond_test = ONDTest(
            algorithm_attribute,
            "",
            DOMAIN,
            FEEDBACK_TYPE,
            tempdirectory,
            ond_harness_instance,
            tempdirectory,
            session_id,
            [],
            True,
            True,
        )
        ond_test(test_ids[0])
