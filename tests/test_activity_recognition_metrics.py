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
    detection_file = os.path.join(result_folder, "OND.10.90001.2100554_PreComputedDetector_detection.csv")
    detection = pd.read_csv(detection_file, sep=",", header=None)
    return detection, gt


@pytest.fixture(scope="function")
def classification_file():
    """Fixture for reading classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.10.90001.2100554_PreComputedDetector_classification.csv"
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
        result_folder, "OND.10.90001.2100554_BaselinePreComputedDetector_classification.csv"
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


def test_m_acc(arm_metrics, detection_files, classification_file):
    """
    Test m_acc computation.

    Args:
        program_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    m_acc = arm_metrics.m_acc(gt[1], classification_file, gt[3], 100, 5)
    assert m_acc == {
        "full_top1": 0.1988,
        "full_top3": 0.25602,
        "post_top1": 0.30556,
        "post_top3": 0.34722,
        "pre_top1": 0.0,
        "pre_top3": 0.08621,
    }


def test_m_num(arm_metrics, detection_files):
    """
    Test m_num computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num = arm_metrics.m_num(detection[arm_metrics.novel_id], gt[1])
    assert m_num == 24


def test_m_num_stats(arm_metrics, detection_files):
    """
    Test m_num_stats computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = arm_metrics.m_num_stats(detection[arm_metrics.novel_id], gt[1])
    assert m_num_stats["GT_indx"] == 117 and m_num_stats["P_indx"] == 193


def test_m_ndp(arm_metrics, detection_files):
    """
    Test m_ndp computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp = arm_metrics.m_ndp(detection[arm_metrics.novel_id], gt[1])
    assert m_ndp == {
        "FN": 23,
        "FP": 65,
        "TN": 169,
        "TP": 75,
        "accuracy": 0.73494,
        "f1_score": 0.63025,
        "precision": 0.53571,
        "recall": 0.76531,
    }


def test_m_ndp_pre(arm_metrics, detection_files):
    """
    Test m_ndp_pre computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_pre = arm_metrics.m_ndp_pre(detection[arm_metrics.novel_id], gt[1])
    assert m_ndp_pre == {
        "FN": 0,
        "FP": 0,
        "TN": 116,
        "TP": 0,
        "accuracy": 1.0,
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }


def test_m_ndp_post(arm_metrics, detection_files):
    """
    Test m_ndp_post computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_post = arm_metrics.m_ndp_post(detection[arm_metrics.novel_id], gt[1])
    assert m_ndp_post == {
        "FN": 23,
        "FP": 65,
        "TN": 53,
        "TP": 75,
        "accuracy": 0.59259,
        "f1_score": 0.63025,
        "precision": 0.53571,
        "recall": 0.76531,
    }


def test_m_ndp_failed_reaction(arm_metrics, detection_files, classification_file):
    """
    Test m_ndp_failed_reaction computation.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_failed = arm_metrics.m_ndp_failed_reaction(
        detection[arm_metrics.novel_id], gt[1], classification_file, gt[3]
    )
    assert m_ndp_failed == {
        "top1_FN": 8,
        "top1_FP": 65,
        "top1_TN": 169,
        "top1_TP": 24,
        "top1_accuracy": 0.72556,
        "top1_f1_score": 0.39669,
        "top1_precision": 0.26966,
        "top1_recall": 0.75,
        "top3_FN": 8,
        "top3_FP": 58,
        "top3_TN": 157,
        "top3_TP": 24,
        "top3_accuracy": 0.73279,
        "top3_f1_score": 0.42105,
        "top3_precision": 0.29268,
        "top3_recall": 0.75,
    }


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
        m_num_stats["GT_indx"], m_num_stats["P_indx"], gt.shape[0]
    )
    assert is_cdt_is_early["Is CDT"] and not is_cdt_is_early["Is Early"]


def test_m_nrp(
    arm_metrics, detection_files, classification_file, baseline_classification_file
):
    """
    Test novelty reaction performance.

    Args:
        arm_metrics (ActivityRecognitionMetrics): An instance of ActivityRecognitionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file (DataFrame): A data frame with classification prediction
        baseline_classification_fDataFrame): A data frame with classification prediction for baseline

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
    assert m_nrp == {
        "M_nrp_post_top1": 31.092026537506616,
        "M_nrp_post_top3": 35.02390607032621,
    }
