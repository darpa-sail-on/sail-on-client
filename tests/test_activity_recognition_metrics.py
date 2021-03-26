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
    detection_file = os.path.join(result_folder, "OND.10.90001.2100554_detection.csv")
    detection = pd.read_csv(detection_file, sep=",", header=None)
    return detection, gt


@pytest.fixture(scope="function")
def classification_file():
    """Fixture for reading classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "activity_recognition"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.10.90001.2100554_classification.csv"
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
    assert m_num == {
        "0.175": 14,
        "0.225": 14,
        "0.3": 20,
        "0.4": 24,
        "0.5": 24,
        "0.6": 39,
    }


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
    assert m_num_stats["GT_indx"] == 117
    assert m_num_stats["P_indx_0.175"] == 161
    assert m_num_stats["P_indx_0.225"] == 161
    assert m_num_stats["P_indx_0.3"] == 182
    assert m_num_stats["P_indx_0.4"] == 193
    assert m_num_stats["P_indx_0.5"] == 193
    assert m_num_stats["P_indx_0.6"] == 225


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
        "accuracy_0.175": 0.6988,
        "precision_0.175": 0.49419,
        "recall_0.175": 0.86735,
        "f1_score_0.175": 0.62963,
        "TP_0.175": 85,
        "FP_0.175": 87,
        "TN_0.175": 147,
        "FN_0.175": 13,
        "accuracy_0.225": 0.6988,
        "precision_0.225": 0.49419,
        "recall_0.225": 0.86735,
        "f1_score_0.225": 0.62963,
        "TP_0.225": 85,
        "FP_0.225": 87,
        "TN_0.225": 147,
        "FN_0.225": 13,
        "accuracy_0.3": 0.7259,
        "precision_0.3": 0.52318,
        "recall_0.3": 0.80612,
        "f1_score_0.3": 0.63454,
        "TP_0.3": 79,
        "FP_0.3": 72,
        "TN_0.3": 162,
        "FN_0.3": 19,
        "accuracy_0.4": 0.73494,
        "precision_0.4": 0.53571,
        "recall_0.4": 0.76531,
        "f1_score_0.4": 0.63025,
        "TP_0.4": 75,
        "FP_0.4": 65,
        "TN_0.4": 169,
        "FN_0.4": 23,
        "accuracy_0.5": 0.73494,
        "precision_0.5": 0.53571,
        "recall_0.5": 0.76531,
        "f1_score_0.5": 0.63025,
        "TP_0.5": 75,
        "FP_0.5": 65,
        "TN_0.5": 169,
        "FN_0.5": 23,
        "accuracy_0.6": 0.74096,
        "precision_0.6": 0.55556,
        "recall_0.6": 0.61224,
        "f1_score_0.6": 0.58252,
        "TP_0.6": 60,
        "FP_0.6": 48,
        "TN_0.6": 186,
        "FN_0.6": 38,
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
        "accuracy_0.175": 1.0,
        "precision_0.175": 0.0,
        "recall_0.175": 0.0,
        "f1_score_0.175": 0.0,
        "TP_0.175": 0,
        "FP_0.175": 0,
        "TN_0.175": 116,
        "FN_0.175": 0,
        "accuracy_0.225": 1.0,
        "precision_0.225": 0.0,
        "recall_0.225": 0.0,
        "f1_score_0.225": 0.0,
        "TP_0.225": 0,
        "FP_0.225": 0,
        "TN_0.225": 116,
        "FN_0.225": 0,
        "accuracy_0.3": 1.0,
        "precision_0.3": 0.0,
        "recall_0.3": 0.0,
        "f1_score_0.3": 0.0,
        "TP_0.3": 0,
        "FP_0.3": 0,
        "TN_0.3": 116,
        "FN_0.3": 0,
        "accuracy_0.4": 1.0,
        "precision_0.4": 0.0,
        "recall_0.4": 0.0,
        "f1_score_0.4": 0.0,
        "TP_0.4": 0,
        "FP_0.4": 0,
        "TN_0.4": 116,
        "FN_0.4": 0,
        "accuracy_0.5": 1.0,
        "precision_0.5": 0.0,
        "recall_0.5": 0.0,
        "f1_score_0.5": 0.0,
        "TP_0.5": 0,
        "FP_0.5": 0,
        "TN_0.5": 116,
        "FN_0.5": 0,
        "accuracy_0.6": 1.0,
        "precision_0.6": 0.0,
        "recall_0.6": 0.0,
        "f1_score_0.6": 0.0,
        "TP_0.6": 0,
        "FP_0.6": 0,
        "TN_0.6": 116,
        "FN_0.6": 0,
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
        "accuracy_0.175": 0.53704,
        "precision_0.175": 0.49419,
        "recall_0.175": 0.86735,
        "f1_score_0.175": 0.62963,
        "TP_0.175": 85,
        "FP_0.175": 87,
        "TN_0.175": 31,
        "FN_0.175": 13,
        "accuracy_0.225": 0.53704,
        "precision_0.225": 0.49419,
        "recall_0.225": 0.86735,
        "f1_score_0.225": 0.62963,
        "TP_0.225": 85,
        "FP_0.225": 87,
        "TN_0.225": 31,
        "FN_0.225": 13,
        "accuracy_0.3": 0.5787,
        "precision_0.3": 0.52318,
        "recall_0.3": 0.80612,
        "f1_score_0.3": 0.63454,
        "TP_0.3": 79,
        "FP_0.3": 72,
        "TN_0.3": 46,
        "FN_0.3": 19,
        "accuracy_0.4": 0.59259,
        "precision_0.4": 0.53571,
        "recall_0.4": 0.76531,
        "f1_score_0.4": 0.63025,
        "TP_0.4": 75,
        "FP_0.4": 65,
        "TN_0.4": 53,
        "FN_0.4": 23,
        "accuracy_0.5": 0.59259,
        "precision_0.5": 0.53571,
        "recall_0.5": 0.76531,
        "f1_score_0.5": 0.63025,
        "TP_0.5": 75,
        "FP_0.5": 65,
        "TN_0.5": 53,
        "FN_0.5": 23,
        "accuracy_0.6": 0.60185,
        "precision_0.6": 0.55556,
        "recall_0.6": 0.61224,
        "f1_score_0.6": 0.58252,
        "TP_0.6": 60,
        "FP_0.6": 48,
        "TN_0.6": 70,
        "FN_0.6": 38,
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
        "top1_accuracy_0.175": 0.65789,
        "top1_precision_0.175": 0.24348,
        "top1_recall_0.175": 0.875,
        "top1_f1_score_0.175": 0.38095,
        "top1_TP_0.175": 28,
        "top1_FP_0.175": 87,
        "top1_TN_0.175": 147,
        "top1_FN_0.175": 4,
        "top1_accuracy_0.225": 0.65789,
        "top1_precision_0.225": 0.24348,
        "top1_recall_0.225": 0.875,
        "top1_f1_score_0.225": 0.38095,
        "top1_TP_0.225": 28,
        "top1_FP_0.225": 87,
        "top1_TN_0.225": 147,
        "top1_FN_0.225": 4,
        "top1_accuracy_0.3": 0.70301,
        "top1_precision_0.3": 0.25773,
        "top1_recall_0.3": 0.78125,
        "top1_f1_score_0.3": 0.3876,
        "top1_TP_0.3": 25,
        "top1_FP_0.3": 72,
        "top1_TN_0.3": 162,
        "top1_FN_0.3": 7,
        "top1_accuracy_0.4": 0.72556,
        "top1_precision_0.4": 0.26966,
        "top1_recall_0.4": 0.75,
        "top1_f1_score_0.4": 0.39669,
        "top1_TP_0.4": 24,
        "top1_FP_0.4": 65,
        "top1_TN_0.4": 169,
        "top1_FN_0.4": 8,
        "top1_accuracy_0.5": 0.72556,
        "top1_precision_0.5": 0.26966,
        "top1_recall_0.5": 0.75,
        "top1_f1_score_0.5": 0.39669,
        "top1_TP_0.5": 24,
        "top1_FP_0.5": 65,
        "top1_TN_0.5": 169,
        "top1_FN_0.5": 8,
        "top1_accuracy_0.6": 0.76692,
        "top1_precision_0.6": 0.27273,
        "top1_recall_0.6": 0.5625,
        "top1_f1_score_0.6": 0.36735,
        "top1_TP_0.6": 18,
        "top1_FP_0.6": 48,
        "top1_TN_0.6": 186,
        "top1_FN_0.6": 14,
        "top3_accuracy_0.175": 0.66397,
        "top3_precision_0.175": 0.26168,
        "top3_recall_0.175": 0.875,
        "top3_f1_score_0.175": 0.40288,
        "top3_TP_0.175": 28,
        "top3_FP_0.175": 79,
        "top3_TN_0.175": 136,
        "top3_FN_0.175": 4,
        "top3_accuracy_0.225": 0.66397,
        "top3_precision_0.225": 0.26168,
        "top3_recall_0.225": 0.875,
        "top3_f1_score_0.225": 0.40288,
        "top3_TP_0.225": 28,
        "top3_FP_0.225": 79,
        "top3_TN_0.225": 136,
        "top3_FN_0.225": 4,
        "top3_accuracy_0.3": 0.7085,
        "top3_precision_0.3": 0.27778,
        "top3_recall_0.3": 0.78125,
        "top3_f1_score_0.3": 0.40984,
        "top3_TP_0.3": 25,
        "top3_FP_0.3": 65,
        "top3_TN_0.3": 150,
        "top3_FN_0.3": 7,
        "top3_accuracy_0.4": 0.73279,
        "top3_precision_0.4": 0.29268,
        "top3_recall_0.4": 0.75,
        "top3_f1_score_0.4": 0.42105,
        "top3_TP_0.4": 24,
        "top3_FP_0.4": 58,
        "top3_TN_0.4": 157,
        "top3_FN_0.4": 8,
        "top3_accuracy_0.5": 0.73279,
        "top3_precision_0.5": 0.29268,
        "top3_recall_0.5": 0.75,
        "top3_f1_score_0.5": 0.42105,
        "top3_TP_0.5": 24,
        "top3_FP_0.5": 58,
        "top3_TN_0.5": 157,
        "top3_FN_0.5": 8,
        "top3_accuracy_0.6": 0.77733,
        "top3_precision_0.6": 0.30508,
        "top3_recall_0.6": 0.5625,
        "top3_f1_score_0.6": 0.3956,
        "top3_TP_0.6": 18,
        "top3_FP_0.6": 41,
        "top3_TN_0.6": 174,
        "top3_FN_0.6": 14,
    }


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
