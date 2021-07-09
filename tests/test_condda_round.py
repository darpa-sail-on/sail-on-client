"""Tests for CONDDARound."""

import numpy as np
import pytest

from sail_on_client.protocol.condda_round import CONDDARound


def test_initialize(harness_instance, algorithm_instance):
    """
    Test initialization of CONDDARound.

    Args:
        harness_instance: Instance of local interface
        algorithm_instance: Instance of PreComputedDetector

    Returns:
        None
    """
    data_root = ""
    features_dict = {}
    logit_dict = {}
    redlight_instance = ""
    test_ids = ["CONDDA.10.90001.2100554"]
    session_id = harness_instance.session_request(test_ids,
                                                  "CONDDA",
                                                  "activity_recognition",
                                                  "0.0.0",
                                                  [],
                                                  0.5)
    CONDDARound(algorithm_instance, data_root, features_dict, harness_instance,
                logit_dict, redlight_instance, session_id, [], test_ids[0])


def test_call(harness_instance, algorithm_instance):
    """
    Test __call__ of CONDDARound.

    Args:
        harness_instance: Instance of local interface
        algorithm_instance: Instance of PreComputedDetector

    Returns:
        None
    """
    data_root = ""
    redlight_instance = ""
    test_ids = ["CONDDA.10.90001.2100554"]
    session_id = harness_instance.session_request(test_ids,
                                                  "CONDDA",
                                                  "activity_recognition",
                                                  "0.0.0",
                                                  [],
                                                  0.5)
    dataset = harness_instance.dataset_request(test_ids[0], 0, session_id)
    # Run a round without features
    features_dict = {}
    logit_dict = {}
    algorithm_instance.execute({"test_id": test_ids[0]}, "Initialize")
    condda_round = CONDDARound(algorithm_instance, data_root, features_dict,
                               harness_instance, logit_dict, redlight_instance,
                               session_id, [], test_ids[0])
    condda_round(dataset, 0)


def test_call_with_features(harness_instance, algorithm_instance):
    """
    Test __call__ of CONDDARound with features.

    Args:
        harness_instance: Instance of local interface
        algorithm_instance: Instance of PreComputedDetector

    Returns:
        None
    """
    data_root = ""
    redlight_instance = ""
    test_ids = ["CONDDA.10.90001.2100554"]
    session_id = harness_instance.session_request(test_ids,
                                                  "CONDDA",
                                                  "activity_recognition",
                                                  "0.0.0",
                                                  [],
                                                  0.5)
    dataset = harness_instance.dataset_request(test_ids[0], 0, session_id)
    # Run a round without features
    algorithm_instance.execute({"test_id": test_ids[0]}, "Initialize")
    # Run a round with features
    features_dict = {}
    logit_dict = {}
    instance_ids = CONDDARound.get_instance_ids(dataset)
    for instance_id in instance_ids:
        features_dict[instance_id] = np.random.random(1024)
        logit_dict[instance_id] = np.random.random(88)

    algorithm_instance.execute({"test_id": test_ids[0]}, "Initialize")
    condda_round_with_features = CONDDARound(algorithm_instance, data_root, features_dict,
                                             harness_instance, logit_dict, redlight_instance,
                                             session_id, [], test_ids[0])
    condda_round_with_features(dataset, 0)