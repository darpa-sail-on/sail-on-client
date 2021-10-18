"""Tests for ONDRound."""

import numpy as np

from sail_on_client.protocol.ond_round import ONDRound


def test_initialize(ond_harness_instance, ond_algorithm_instance):
    """
    Test initialization of ONDRound.

    Args:
        ond_harness_instance: Instance of local interface
        ond_algorithm_instance: Instance of PreComputedONDAgent

    Returns:
        None
    """
    data_root = ""
    features_dict = {}
    logit_dict = {}
    redlight_instance = ""
    test_ids = ["OND.10.90001.2100554"]
    session_id = ond_harness_instance.session_request(
        test_ids, "OND", "activity_recognition", "0.0.0", [], 0.5
    )
    ONDRound(
        ond_algorithm_instance,
        data_root,
        features_dict,
        ond_harness_instance,
        logit_dict,
        redlight_instance,
        session_id,
        [],
        test_ids[0],
    )


def test_call(ond_harness_instance, ond_algorithm_instance):
    """
    Test __call__ of ONDRound.

    Args:
        ond_harness_instance: Instance of local interface
        ond_algorithm_instance: Instance of PreComputedONDAgent

    Returns:
        None
    """
    data_root = ""
    redlight_instance = ""
    test_ids = ["OND.10.90001.2100554"]
    session_id = ond_harness_instance.session_request(
        test_ids, "OND", "activity_recognition", "0.0.0", [], 0.5
    )
    dataset = ond_harness_instance.dataset_request(test_ids[0], 0, session_id)
    # Run a round without features
    features_dict = {}
    logit_dict = {}
    ond_algorithm_instance.execute({"test_id": test_ids[0]}, "Initialize")
    ond_round = ONDRound(
        ond_algorithm_instance,
        data_root,
        features_dict,
        ond_harness_instance,
        logit_dict,
        redlight_instance,
        session_id,
        [],
        test_ids[0],
    )
    ond_round(dataset, 0)


def test_call_with_features(ond_harness_instance, ond_algorithm_instance):
    """
    Test __call__ of ONDRound with features.

    Args:
        ond_harness_instance: Instance of local interface
        ond_algorithm_instance: Instance of PreComputedONDAgent

    Returns:
        None
    """
    data_root = ""
    redlight_instance = ""
    test_ids = ["OND.10.90001.2100554"]
    session_id = ond_harness_instance.session_request(
        test_ids, "OND", "activity_recognition", "0.0.0", [], 0.5
    )
    dataset = ond_harness_instance.dataset_request(test_ids[0], 0, session_id)
    # Run a round without features
    ond_algorithm_instance.execute({"test_id": test_ids[0]}, "Initialize")
    # Run a round with features
    features_dict = {}
    logit_dict = {}
    instance_ids = ONDRound.get_instance_ids(dataset)
    for instance_id in instance_ids:
        features_dict[instance_id] = np.random.random(1024)
        logit_dict[instance_id] = np.random.random(88)

    ond_algorithm_instance.execute({"test_id": test_ids[0]}, "Initialize")
    ond_round_with_features = ONDRound(
        ond_algorithm_instance,
        data_root,
        features_dict,
        ond_harness_instance,
        logit_dict,
        redlight_instance,
        session_id,
        [],
        test_ids[0],
    )
    ond_round_with_features(dataset, 0)
