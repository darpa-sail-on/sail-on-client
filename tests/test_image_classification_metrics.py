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
    detection_file = os.path.join(
        result_folder, "OND.54011215.0000.1236_PreComputedDetector_detection.csv"
    )
    detection = pd.read_csv(detection_file, sep=",", header=None)
    return detection, gt


@pytest.fixture(scope="function")
def classification_file():
    """Fixture for reading classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.54011215.0000.1236_PreComputedDetector_classification.csv"
    )
    classification = pd.read_csv(classification_file_id, sep=",", header=None)
    return classification


@pytest.fixture(scope="function")
def baseline_classification_file():
    """Fixture for reading baseline classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "image_classification"
    )
    classification_file_id = os.path.join(
        result_folder,
        "OND.54011215.0000.1236_BaselinePreComputedDetector_classification.csv",
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
        "full_top1": 0.01016,
        "full_top3": 0.05547,
        "pre_top1": 0.01463,
        "pre_top3": 0.07495,
        "post_top1": 0.00894,
        "post_top3": 0.05017,
        "asymptotic_500_top1": 0.02,
        "asymptotic_500_top3": 0.064,
        "asymptotic_600_top1": 0.01667,
        "asymptotic_600_top3": 0.06333,
        "asymptotic_700_top1": 0.01571,
        "asymptotic_700_top3": 0.06143,
        "asymptotic_800_top1": 0.015,
        "asymptotic_800_top3": 0.0575,
        "asymptotic_900_top1": 0.01333,
        "asymptotic_900_top3": 0.05667,
        "asymptotic_1000_top1": 0.012,
        "asymptotic_1000_top3": 0.058,
        "asymptotic_1100_top1": 0.01091,
        "asymptotic_1100_top3": 0.05636,
        "asymptotic_1200_top1": 0.01083,
        "asymptotic_1200_top3": 0.0575,
        "asymptotic_1300_top1": 0.01077,
        "asymptotic_1300_top3": 0.05615,
        "asymptotic_1400_top1": 0.01143,
        "asymptotic_1400_top3": 0.055,
        "asymptotic_1500_top1": 0.01067,
        "asymptotic_1500_top3": 0.05333,
        "asymptotic_1600_top1": 0.01063,
        "asymptotic_1600_top3": 0.0525,
        "asymptotic_1700_top1": 0.01,
        "asymptotic_1700_top3": 0.05118,
        "asymptotic_1800_top1": 0.00944,
        "asymptotic_1800_top3": 0.05222,
        "asymptotic_1900_top1": 0.00947,
        "asymptotic_1900_top3": 0.05211,
        "asymptotic_2000_top1": 0.009,
        "asymptotic_2000_top3": 0.0505,
        "asymptotic_2100_top1": 0.00952,
        "asymptotic_2100_top3": 0.05143,
        "asymptotic_2200_top1": 0.01091,
        "asymptotic_2200_top3": 0.05227,
        "asymptotic_2300_top1": 0.01087,
        "asymptotic_2300_top3": 0.05217,
        "asymptotic_2400_top1": 0.01042,
        "asymptotic_2400_top3": 0.05458,
        "asymptotic_2500_top1": 0.0104,
        "asymptotic_2500_top3": 0.0556,
    }


def test_m_acc_round_wise(arm_metrics, detection_files, classification_file):
    """
    Test m_acc computation for a round.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    _, gt = detection_files
    m_acc_round_wise = arm_metrics.m_acc_round_wise(
        classification_file,
        gt[arm_metrics.classification_id],
        0
    )
    assert m_acc_round_wise == {
        "top1_accuracy_round_0": 0.01016,
        "top3_accuracy_round_0": 0.05547
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
    assert m_num == {"0.175": 1, "0.225": 1, "0.3": 1, "0.4": 1, "0.5": 1, "0.6": 1}


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
    assert m_num_stats == {
        "GT_indx": 548,
        "P_indx_0.175": 548,
        "P_indx_0.225": 548,
        "P_indx_0.3": 548,
        "P_indx_0.4": 548,
        "P_indx_0.5": 548,
        "P_indx_0.6": 548,
    }


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
        "accuracy_0.175": 1.0,
        "precision_0.175": 1.0,
        "recall_0.175": 1.0,
        "f1_score_0.175": 1.0,
        "TP_0.175": 2013,
        "FP_0.175": 0,
        "TN_0.175": 547,
        "FN_0.175": 0,
        "accuracy_0.225": 1.0,
        "precision_0.225": 1.0,
        "recall_0.225": 1.0,
        "f1_score_0.225": 1.0,
        "TP_0.225": 2013,
        "FP_0.225": 0,
        "TN_0.225": 547,
        "FN_0.225": 0,
        "accuracy_0.3": 1.0,
        "precision_0.3": 1.0,
        "recall_0.3": 1.0,
        "f1_score_0.3": 1.0,
        "TP_0.3": 2013,
        "FP_0.3": 0,
        "TN_0.3": 547,
        "FN_0.3": 0,
        "accuracy_0.4": 1.0,
        "precision_0.4": 1.0,
        "recall_0.4": 1.0,
        "f1_score_0.4": 1.0,
        "TP_0.4": 2013,
        "FP_0.4": 0,
        "TN_0.4": 547,
        "FN_0.4": 0,
        "accuracy_0.5": 1.0,
        "precision_0.5": 1.0,
        "recall_0.5": 1.0,
        "f1_score_0.5": 1.0,
        "TP_0.5": 2013,
        "FP_0.5": 0,
        "TN_0.5": 547,
        "FN_0.5": 0,
        "accuracy_0.6": 1.0,
        "precision_0.6": 1.0,
        "recall_0.6": 1.0,
        "f1_score_0.6": 1.0,
        "TP_0.6": 2013,
        "FP_0.6": 0,
        "TN_0.6": 547,
        "FN_0.6": 0,
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
        "accuracy_0.175": 1.0,
        "precision_0.175": 0.0,
        "recall_0.175": 0.0,
        "f1_score_0.175": 0.0,
        "TP_0.175": 0,
        "FP_0.175": 0,
        "TN_0.175": 547,
        "FN_0.175": 0,
        "accuracy_0.225": 1.0,
        "precision_0.225": 0.0,
        "recall_0.225": 0.0,
        "f1_score_0.225": 0.0,
        "TP_0.225": 0,
        "FP_0.225": 0,
        "TN_0.225": 547,
        "FN_0.225": 0,
        "accuracy_0.3": 1.0,
        "precision_0.3": 0.0,
        "recall_0.3": 0.0,
        "f1_score_0.3": 0.0,
        "TP_0.3": 0,
        "FP_0.3": 0,
        "TN_0.3": 547,
        "FN_0.3": 0,
        "accuracy_0.4": 1.0,
        "precision_0.4": 0.0,
        "recall_0.4": 0.0,
        "f1_score_0.4": 0.0,
        "TP_0.4": 0,
        "FP_0.4": 0,
        "TN_0.4": 547,
        "FN_0.4": 0,
        "accuracy_0.5": 1.0,
        "precision_0.5": 0.0,
        "recall_0.5": 0.0,
        "f1_score_0.5": 0.0,
        "TP_0.5": 0,
        "FP_0.5": 0,
        "TN_0.5": 547,
        "FN_0.5": 0,
        "accuracy_0.6": 1.0,
        "precision_0.6": 0.0,
        "recall_0.6": 0.0,
        "f1_score_0.6": 0.0,
        "TP_0.6": 0,
        "FP_0.6": 0,
        "TN_0.6": 547,
        "FN_0.6": 0,
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
        "accuracy_0.175": 1.0,
        "precision_0.175": 1.0,
        "recall_0.175": 1.0,
        "f1_score_0.175": 1.0,
        "TP_0.175": 2013,
        "FP_0.175": 0,
        "TN_0.175": 0,
        "FN_0.175": 0,
        "accuracy_0.225": 1.0,
        "precision_0.225": 1.0,
        "recall_0.225": 1.0,
        "f1_score_0.225": 1.0,
        "TP_0.225": 2013,
        "FP_0.225": 0,
        "TN_0.225": 0,
        "FN_0.225": 0,
        "accuracy_0.3": 1.0,
        "precision_0.3": 1.0,
        "recall_0.3": 1.0,
        "f1_score_0.3": 1.0,
        "TP_0.3": 2013,
        "FP_0.3": 0,
        "TN_0.3": 0,
        "FN_0.3": 0,
        "accuracy_0.4": 1.0,
        "precision_0.4": 1.0,
        "recall_0.4": 1.0,
        "f1_score_0.4": 1.0,
        "TP_0.4": 2013,
        "FP_0.4": 0,
        "TN_0.4": 0,
        "FN_0.4": 0,
        "accuracy_0.5": 1.0,
        "precision_0.5": 1.0,
        "recall_0.5": 1.0,
        "f1_score_0.5": 1.0,
        "TP_0.5": 2013,
        "FP_0.5": 0,
        "TN_0.5": 0,
        "FN_0.5": 0,
        "accuracy_0.6": 1.0,
        "precision_0.6": 1.0,
        "recall_0.6": 1.0,
        "f1_score_0.6": 1.0,
        "TP_0.6": 2013,
        "FP_0.6": 0,
        "TN_0.6": 0,
        "FN_0.6": 0,
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
        "top1_accuracy_0.175": 1.0,
        "top1_precision_0.175": 1.0,
        "top1_recall_0.175": 1.0,
        "top1_f1_score_0.175": 1.0,
        "top1_TP_0.175": 1995,
        "top1_FP_0.175": 0,
        "top1_TN_0.175": 539,
        "top1_FN_0.175": 0,
        "top1_accuracy_0.225": 1.0,
        "top1_precision_0.225": 1.0,
        "top1_recall_0.225": 1.0,
        "top1_f1_score_0.225": 1.0,
        "top1_TP_0.225": 1995,
        "top1_FP_0.225": 0,
        "top1_TN_0.225": 539,
        "top1_FN_0.225": 0,
        "top1_accuracy_0.3": 1.0,
        "top1_precision_0.3": 1.0,
        "top1_recall_0.3": 1.0,
        "top1_f1_score_0.3": 1.0,
        "top1_TP_0.3": 1995,
        "top1_FP_0.3": 0,
        "top1_TN_0.3": 539,
        "top1_FN_0.3": 0,
        "top1_accuracy_0.4": 1.0,
        "top1_precision_0.4": 1.0,
        "top1_recall_0.4": 1.0,
        "top1_f1_score_0.4": 1.0,
        "top1_TP_0.4": 1995,
        "top1_FP_0.4": 0,
        "top1_TN_0.4": 539,
        "top1_FN_0.4": 0,
        "top1_accuracy_0.5": 1.0,
        "top1_precision_0.5": 1.0,
        "top1_recall_0.5": 1.0,
        "top1_f1_score_0.5": 1.0,
        "top1_TP_0.5": 1995,
        "top1_FP_0.5": 0,
        "top1_TN_0.5": 539,
        "top1_FN_0.5": 0,
        "top1_accuracy_0.6": 1.0,
        "top1_precision_0.6": 1.0,
        "top1_recall_0.6": 1.0,
        "top1_f1_score_0.6": 1.0,
        "top1_TP_0.6": 1995,
        "top1_FP_0.6": 0,
        "top1_TN_0.6": 539,
        "top1_FN_0.6": 0,
        "top3_accuracy_0.175": 1.0,
        "top3_precision_0.175": 1.0,
        "top3_recall_0.175": 1.0,
        "top3_f1_score_0.175": 1.0,
        "top3_TP_0.175": 1912,
        "top3_FP_0.175": 0,
        "top3_TN_0.175": 506,
        "top3_FN_0.175": 0,
        "top3_accuracy_0.225": 1.0,
        "top3_precision_0.225": 1.0,
        "top3_recall_0.225": 1.0,
        "top3_f1_score_0.225": 1.0,
        "top3_TP_0.225": 1912,
        "top3_FP_0.225": 0,
        "top3_TN_0.225": 506,
        "top3_FN_0.225": 0,
        "top3_accuracy_0.3": 1.0,
        "top3_precision_0.3": 1.0,
        "top3_recall_0.3": 1.0,
        "top3_f1_score_0.3": 1.0,
        "top3_TP_0.3": 1912,
        "top3_FP_0.3": 0,
        "top3_TN_0.3": 506,
        "top3_FN_0.3": 0,
        "top3_accuracy_0.4": 1.0,
        "top3_precision_0.4": 1.0,
        "top3_recall_0.4": 1.0,
        "top3_f1_score_0.4": 1.0,
        "top3_TP_0.4": 1912,
        "top3_FP_0.4": 0,
        "top3_TN_0.4": 506,
        "top3_FN_0.4": 0,
        "top3_accuracy_0.5": 1.0,
        "top3_precision_0.5": 1.0,
        "top3_recall_0.5": 1.0,
        "top3_f1_score_0.5": 1.0,
        "top3_TP_0.5": 1912,
        "top3_FP_0.5": 0,
        "top3_TN_0.5": 506,
        "top3_FN_0.5": 0,
        "top3_accuracy_0.6": 1.0,
        "top3_precision_0.6": 1.0,
        "top3_recall_0.6": 1.0,
        "top3_f1_score_0.6": 1.0,
        "top3_TP_0.6": 1912,
        "top3_FP_0.6": 0,
        "top3_TN_0.6": 506,
        "top3_FN_0.6": 0,
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
        "top3_acc_novel_only": 0.05017,
        "top1_acc_novel_only": 0.00894,
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
        m_num_stats["GT_indx"], m_num_stats["P_indx_0.5"], gt.shape[0]
    )
    assert is_cdt_is_early["Is CDT"] and not is_cdt_is_early["Is Early"]


def test_m_nrp(
    arm_metrics, detection_files, classification_file, baseline_classification_file
):
    """
    Test novelty reaction performance.

    Args:
        arm_metrics (ImageClassificationMetrics): An instance of ImageClassificationMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file (DataFrame): A data frame with classification prediction
        baseline_classification_fDataFrame): A data frame with classification prediction for baseline

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
    m_acc_baseline = arm_metrics.m_acc(
        gt[arm_metrics.detection_id],
        baseline_classification_file,
        gt[arm_metrics.classification_id],
        100,
        5,
    )
    m_nrp = arm_metrics.m_nrp(m_acc, m_acc_baseline)
    assert m_nrp == {
        "M_nrp_post_top3": 5.466749479694463,
        "M_nrp_post_top1": 1.1114011859918695,
    }
