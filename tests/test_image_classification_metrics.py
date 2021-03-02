"""Tests for image classification metric class."""

from sail_on_client.evaluate.image_classification import ImageClassificationMetrics

import os
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def arm_metrics():
    """Fixture for generated image classification metrics."""
    gt_config = list(range(0, 3))
    arm_metrics = ImageClassificationMetrics("OND", *gt_config)
    return arm_metrics


@pytest.fixture(scope="function")
def detection_files():
    """Fixture for reading detection file and ground truth."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    gt_file = os.path.join(result_folder, "OND.54011215.0000.1236_single_df.csv")
    gt = pd.read_csv(gt_file, sep=",", header=None, skiprows=1)
    detection_file = os.path.join(result_folder, "OND.54011215.0000.1236_detection.csv")
    detection = pd.read_csv(detection_file, sep=",", header=None)
    return detection, gt


@pytest.fixture(scope="function")
def classification_file():
    """Fixture for reading classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.54011215.0000.1236_classification.csv"
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
    gt_config = list(range(0, 3))
    arm_metrics = ImageClassificationMetrics(protocol_name, *gt_config)
    assert arm_metrics.protocol == protocol_name


def test_m_acc(arm_metrics, detection_files, classification_file):
    """
    Test m_acc computation.

    Args:
        program_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    m_acc = arm_metrics.m_acc(
        gt[arm_metrics.detection_id],
        classification_file,
        gt[arm_metrics.classification_id],
        100,
        5,
    )
    assert m_acc == {
        "asymptotic_1000_top1": 0.501,
        "asymptotic_1000_top3": 0.501,
        "asymptotic_1100_top1": 0.49727,
        "asymptotic_1100_top3": 0.49727,
        "asymptotic_1200_top1": 0.49917,
        "asymptotic_1200_top3": 0.49917,
        "asymptotic_1300_top1": 0.49692,
        "asymptotic_1300_top3": 0.49692,
        "asymptotic_1400_top1": 0.49071,
        "asymptotic_1400_top3": 0.49071,
        "asymptotic_1500_top1": 0.49067,
        "asymptotic_1500_top3": 0.49067,
        "asymptotic_1600_top1": 0.49188,
        "asymptotic_1600_top3": 0.49188,
        "asymptotic_1700_top1": 0.49,
        "asymptotic_1700_top3": 0.49,
        "asymptotic_1800_top1": 0.49167,
        "asymptotic_1800_top3": 0.49167,
        "asymptotic_1900_top1": 0.49421,
        "asymptotic_1900_top3": 0.49421,
        "asymptotic_2000_top1": 0.4915,
        "asymptotic_2000_top3": 0.4915,
        "asymptotic_2100_top1": 0.51143,
        "asymptotic_2100_top3": 0.51143,
        "asymptotic_2200_top1": 0.53364,
        "asymptotic_2200_top3": 0.53364,
        "asymptotic_2300_top1": 0.55391,
        "asymptotic_2300_top3": 0.55391,
        "asymptotic_2400_top1": 0.5725,
        "asymptotic_2400_top3": 0.5725,
        "asymptotic_2500_top1": 0.5896,
        "asymptotic_2500_top3": 0.5896,
        "asymptotic_500_top1": 0.492,
        "asymptotic_500_top3": 0.492,
        "asymptotic_600_top1": 0.49167,
        "asymptotic_600_top3": 0.49167,
        "asymptotic_700_top1": 0.48714,
        "asymptotic_700_top3": 0.48714,
        "asymptotic_800_top1": 0.49125,
        "asymptotic_800_top3": 0.49125,
        "asymptotic_900_top1": 0.49556,
        "asymptotic_900_top3": 0.49556,
        "full_top1": 0.59922,
        "full_top3": 0.59922,
        "post_top1": 0.49031,
        "post_top3": 0.49031,
        "pre_top1": 1.0,
        "pre_top3": 1.0,
    }


def test_m_num(arm_metrics, detection_files):
    """
    Test m_num computation.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num = arm_metrics.m_num(detection[1], gt[arm_metrics.detection_id])
    assert m_num == 1


def test_m_num_stats(arm_metrics, detection_files):
    """
    Test m_num_stats computation.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = arm_metrics.m_num_stats(detection[1], gt[arm_metrics.detection_id])
    assert m_num_stats["GT_indx"] == 548 and m_num_stats["P_indx"] == 548


def test_m_ndp(arm_metrics, detection_files):
    """
    Test m_ndp computation.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp = arm_metrics.m_ndp(detection[1], gt[arm_metrics.detection_id])
    assert m_ndp == {
        "FN": 0,
        "FP": 0,
        "TN": 547,
        "TP": 2013,
        "accuracy": 1.0,
        "f1_score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }


def test_m_ndp_pre(arm_metrics, detection_files):
    """
    Test m_ndp_pre computation.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_pre = arm_metrics.m_ndp_pre(detection[1], gt[arm_metrics.detection_id])
    assert m_ndp_pre == {
        "FN": 0,
        "FP": 0,
        "TN": 547,
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
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_post = arm_metrics.m_ndp_post(detection[1], gt[arm_metrics.detection_id])
    assert m_ndp_post == {
        "FN": 0,
        "FP": 0,
        "TN": 0,
        "TP": 2013,
        "accuracy": 1.0,
        "f1_score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }


# @pytest.mark.skip(reason="no way of currently testing this since no gt_class")
def test_m_ndp_failed_reaction(arm_metrics, detection_files, classification_file):
    """
    Test m_ndp_failed_reaction computation.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_failed = arm_metrics.m_ndp_failed_reaction(
        detection[1],
        gt[arm_metrics.detection_id],
        classification_file,
        gt[arm_metrics.classification_id],
    )
    assert m_ndp_failed == {
        "top1_FN": 0,
        "top1_FP": 0,
        "top1_TN": 0,
        "top1_TP": 1026,
        "top1_accuracy": 1.0,
        "top1_f1_score": 1.0,
        "top1_precision": 1.0,
        "top1_recall": 1.0,
        "top3_FN": 0,
        "top3_FP": 0,
        "top3_TN": 0,
        "top3_TP": 1026,
        "top3_accuracy": 1.0,
        "top3_f1_score": 1.0,
        "top3_precision": 1.0,
        "top3_recall": 1.0,
    }


def test_m_accuracy_on_novel(arm_metrics, detection_files, classification_file):
    """
    Test m_accuracy_on_novel computation.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    m_accuracy_on_novel = arm_metrics.m_accuracy_on_novel(
        classification_file,
        gt[arm_metrics.classification_id],
        gt[arm_metrics.detection_id],
    )

    assert m_accuracy_on_novel == {
        "top1_acc_novel_only": 0.49031,
        "top3_acc_novel_only": 0.49031,
    }


def test_is_cdt_and_is_early(arm_metrics, detection_files):
    """
    Test m_is_cdt_and_is_early computation.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = arm_metrics.m_num_stats(detection[1], gt[arm_metrics.detection_id])
    is_cdt_is_early = arm_metrics.m_is_cdt_and_is_early(
        m_num_stats["GT_indx"], m_num_stats["P_indx"], gt.shape[0]
    )
    assert is_cdt_is_early["Is CDT"] and not is_cdt_is_early["Is Early"]
