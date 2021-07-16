"""Tests for activity recognition metric class."""

from sail_on_client.evaluate.activity_recognition import ActivityRecognitionMetrics

import os
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def arm_metrics():
    """Fixture for generated activity recognition metrics."""
    gt_config = list(range(0, 6))
    arm_metrics = ActivityRecognitionMetrics("OND", *gt_config)
    return arm_metrics


@pytest.fixture(scope="function")
def detection_files():
    """Fixture for reading detection file and ground truth."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    gt_file = os.path.join(result_folder, "OND.10.90001.2100554_single_df.csv")
    gt = pd.read_csv(gt_file, sep=",", header=None, skiprows=1)
    detection_file = os.path.join(
        result_folder, "OND.10.90001.2100554_PreComputedONDAgent_detection.csv"
    )
    detection = pd.read_csv(detection_file, sep=",", header=None)
    return detection, gt


@pytest.fixture(scope="function")
def classification_file():
    """Fixture for reading classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.10.90001.2100554_PreComputedONDAgent_classification.csv"
    )
    classification = pd.read_csv(classification_file_id, sep=",", header=None)
    return classification


@pytest.fixture(scope="function")
def baseline_classification_file():
    """Fixture for reading baseline classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    classification_file_id = os.path.join(
        result_folder,
        "OND.10.90001.2100554_BaselinePreComputedONDAgent_classification.csv",
    )
    classification = pd.read_csv(classification_file_id, sep=",", header=None)
    return classification


@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_initialize(protocol_name):
    """
    Test activity recognition metric initialization.

    Return:
        None
    """
    gt_config = list(range(0, 6))
    arm_metrics = ActivityRecognitionMetrics(protocol_name, *gt_config)
    assert arm_metrics.protocol == protocol_name


def test_m_acc(
    arm_metrics, detection_files, classification_file, expected_ar_m_acc_values
):
    """
    Test m_acc computation.

    Args:
        program_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_ar_m_acc_values: Values expected_ar for m_acc

    Return:
        None
    """
    detection, gt = detection_files
    m_acc = arm_metrics.m_acc(gt[1], classification_file, gt[3], 100, 5)
    assert m_acc == expected_ar_m_acc_values


def test_m_acc_round_wise(
    arm_metrics,
    detection_files,
    classification_file,
    expected_ar_m_acc_roundwise_values,
):
    """
    Test m_acc computation for a round.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_ar_m_acc_roundwise_values: Values expected_ar for m_acc for a round

    Return:
        None
    """
    _, gt = detection_files
    m_acc_round_wise = arm_metrics.m_acc_round_wise(
        classification_file, gt[arm_metrics.classification_id], 0
    )
    assert m_acc_round_wise == expected_ar_m_acc_roundwise_values


def test_m_num(arm_metrics, detection_files, expected_ar_m_num_values):
    """
    Test m_num computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_num_values:  Values expected_ar for m_num

    Return:
        None
    """
    detection, gt = detection_files
    m_num = arm_metrics.m_num(detection[arm_metrics.novel_id], gt[1])
    assert m_num == expected_ar_m_num_values


def test_m_num_stats(arm_metrics, detection_files, expected_ar_m_num_stats_values):
    """
    Test m_num_stats computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_num_stats_values:  Values expected_ar for m_num_stats

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = arm_metrics.m_num_stats(detection[arm_metrics.novel_id], gt[1])
    assert m_num_stats == expected_ar_m_num_stats_values


def test_m_ndp(arm_metrics, detection_files, expected_ar_m_ndp_values):
    """
    Test m_ndp computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_ndp_values: Values expected_ar for m_ndp

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp = arm_metrics.m_ndp(detection[arm_metrics.novel_id], gt[1])
    assert m_ndp == expected_ar_m_ndp_values


def test_m_ndp_pre(arm_metrics, detection_files, expected_ar_m_ndp_pre_values):
    """
    Test m_ndp_pre computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_ndp_pre_values: Values expected_ar from m_ndp_pre

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_pre = arm_metrics.m_ndp_pre(detection[arm_metrics.novel_id], gt[1])
    assert m_ndp_pre == expected_ar_m_ndp_pre_values


def test_m_ndp_post(arm_metrics, detection_files, expected_ar_m_ndp_post_values):
    """
    Test m_ndp_post computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_ndp_post_values: Values expected_ar for m_ndp_post

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_post = arm_metrics.m_ndp_post(detection[arm_metrics.novel_id], gt[1])
    assert m_ndp_post == expected_ar_m_ndp_post_values


def test_m_ndp_failed_reaction(
    arm_metrics, detection_files, classification_file, expected_ar_m_ndp_failed_values
):
    """
    Test m_ndp_failed_reaction computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_ar_m_ndp_failed_values: Values expected_ar for m_ndp_failed_values

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_failed = arm_metrics.m_ndp_failed_reaction(
        detection[arm_metrics.novel_id], gt[1], classification_file, gt[3]
    )
    assert m_ndp_failed == expected_ar_m_ndp_failed_values


@pytest.mark.skip(reason="no way of currently testing this")
def test_m_accuracy_on_novel(arm_metrics, detection_files, classification_file):
    """
    Test m_accuracy_on_novel computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    arm_metrics.m_accuracy_on_novel(classification_file, gt[3], gt[1])


def test_is_cdt_and_is_early(arm_metrics, detection_files):
    """
    Test m_is_cdt_and_is_early computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = arm_metrics.m_num_stats(detection[arm_metrics.novel_id], gt[1])
    is_cdt_is_early = arm_metrics.m_is_cdt_and_is_early(
        m_num_stats["GT_indx"], m_num_stats["P_indx_0.5"], gt.shape[0]
    )
    assert is_cdt_is_early["Is CDT"] and not is_cdt_is_early["Is Early"]


def test_m_nrp(
    arm_metrics,
    detection_files,
    classification_file,
    baseline_classification_file,
    expected_ar_m_nrp_values,
):
    """
    Test novelty reaction performance.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file (DataFrame): A data frame with classification prediction
        baseline_classification_fDataFrame): A data frame with classification prediction for baseline
        expected_ar_m_nrp_values: Expected Values for m_nrp

    Return:
        None
    """
    detection, gt = detection_files
    m_acc = arm_metrics.m_acc(
        gt[arm_metrics.novel_id],
        classification_file,
        gt[arm_metrics.classification_id],
        100,
        5,
    )
    m_acc_baseline = arm_metrics.m_acc(
        gt[arm_metrics.novel_id],
        baseline_classification_file,
        gt[arm_metrics.classification_id],
        100,
        5,
    )
    m_nrp = arm_metrics.m_nrp(m_acc, m_acc_baseline)
    assert m_nrp == expected_ar_m_nrp_values
