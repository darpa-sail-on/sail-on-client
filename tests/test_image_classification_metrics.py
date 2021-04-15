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
        "full_top1": 0.39922,
        "full_top3": 0.49062,
        "pre_top1": 0.69287,
        "pre_top3": 0.83729,
        "post_top1": 0.31942,
        "post_top3": 0.39642,
        "asymptotic_500_top1": 0.296,
        "asymptotic_500_top3": 0.392,
        "asymptotic_600_top1": 0.305,
        "asymptotic_600_top3": 0.39333,
        "asymptotic_700_top1": 0.30857,
        "asymptotic_700_top3": 0.39143,
        "asymptotic_800_top1": 0.3125,
        "asymptotic_800_top3": 0.39125,
        "asymptotic_900_top1": 0.31556,
        "asymptotic_900_top3": 0.39667,
        "asymptotic_1000_top1": 0.318,
        "asymptotic_1000_top3": 0.402,
        "asymptotic_1100_top1": 0.31636,
        "asymptotic_1100_top3": 0.39636,
        "asymptotic_1200_top1": 0.31833,
        "asymptotic_1200_top3": 0.39583,
        "asymptotic_1300_top1": 0.31385,
        "asymptotic_1300_top3": 0.39,
        "asymptotic_1400_top1": 0.31071,
        "asymptotic_1400_top3": 0.38786,
        "asymptotic_1500_top1": 0.312,
        "asymptotic_1500_top3": 0.388,
        "asymptotic_1600_top1": 0.3175,
        "asymptotic_1600_top3": 0.39313,
        "asymptotic_1700_top1": 0.32059,
        "asymptotic_1700_top3": 0.39353,
        "asymptotic_1800_top1": 0.32444,
        "asymptotic_1800_top3": 0.39667,
        "asymptotic_1900_top1": 0.32368,
        "asymptotic_1900_top3": 0.39737,
        "asymptotic_2000_top1": 0.321,
        "asymptotic_2000_top3": 0.3975,
        "asymptotic_2100_top1": 0.33524,
        "asymptotic_2100_top3": 0.41476,
        "asymptotic_2200_top1": 0.34773,
        "asymptotic_2200_top3": 0.43227,
        "asymptotic_2300_top1": 0.36435,
        "asymptotic_2300_top3": 0.45,
        "asymptotic_2400_top1": 0.37958,
        "asymptotic_2400_top3": 0.4675,
        "asymptotic_2500_top1": 0.3928,
        "asymptotic_2500_top3": 0.4832,
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
    import pdb

    pdb.set_trace()
    assert m_ndp_failed == {
        "top1_accuracy_0.175": 1.0,
        "top1_precision_0.175": 1.0,
        "top1_recall_0.175": 1.0,
        "top1_f1_score_0.175": 1.0,
        "top1_TP_0.175": 1370,
        "top1_FP_0.175": 0,
        "top1_TN_0.175": 168,
        "top1_FN_0.175": 0,
        "top1_accuracy_0.225": 1.0,
        "top1_precision_0.225": 1.0,
        "top1_recall_0.225": 1.0,
        "top1_f1_score_0.225": 1.0,
        "top1_TP_0.225": 1370,
        "top1_FP_0.225": 0,
        "top1_TN_0.225": 168,
        "top1_FN_0.225": 0,
        "top1_accuracy_0.3": 1.0,
        "top1_precision_0.3": 1.0,
        "top1_recall_0.3": 1.0,
        "top1_f1_score_0.3": 1.0,
        "top1_TP_0.3": 1370,
        "top1_FP_0.3": 0,
        "top1_TN_0.3": 168,
        "top1_FN_0.3": 0,
        "top1_accuracy_0.4": 1.0,
        "top1_precision_0.4": 1.0,
        "top1_recall_0.4": 1.0,
        "top1_f1_score_0.4": 1.0,
        "top1_TP_0.4": 1370,
        "top1_FP_0.4": 0,
        "top1_TN_0.4": 168,
        "top1_FN_0.4": 0,
        "top1_accuracy_0.5": 1.0,
        "top1_precision_0.5": 1.0,
        "top1_recall_0.5": 1.0,
        "top1_f1_score_0.5": 1.0,
        "top1_TP_0.5": 1370,
        "top1_FP_0.5": 0,
        "top1_TN_0.5": 168,
        "top1_FN_0.5": 0,
        "top1_accuracy_0.6": 1.0,
        "top1_precision_0.6": 1.0,
        "top1_recall_0.6": 1.0,
        "top1_f1_score_0.6": 1.0,
        "top1_TP_0.6": 1370,
        "top1_FP_0.6": 0,
        "top1_TN_0.6": 168,
        "top1_FN_0.6": 0,
        "top3_accuracy_0.175": 1.0,
        "top3_precision_0.175": 1.0,
        "top3_recall_0.175": 1.0,
        "top3_f1_score_0.175": 1.0,
        "top3_TP_0.175": 1215,
        "top3_FP_0.175": 0,
        "top3_TN_0.175": 89,
        "top3_FN_0.175": 0,
        "top3_accuracy_0.225": 1.0,
        "top3_precision_0.225": 1.0,
        "top3_recall_0.225": 1.0,
        "top3_f1_score_0.225": 1.0,
        "top3_TP_0.225": 1215,
        "top3_FP_0.225": 0,
        "top3_TN_0.225": 89,
        "top3_FN_0.225": 0,
        "top3_accuracy_0.3": 1.0,
        "top3_precision_0.3": 1.0,
        "top3_recall_0.3": 1.0,
        "top3_f1_score_0.3": 1.0,
        "top3_TP_0.3": 1215,
        "top3_FP_0.3": 0,
        "top3_TN_0.3": 89,
        "top3_FN_0.3": 0,
        "top3_accuracy_0.4": 1.0,
        "top3_precision_0.4": 1.0,
        "top3_recall_0.4": 1.0,
        "top3_f1_score_0.4": 1.0,
        "top3_TP_0.4": 1215,
        "top3_FP_0.4": 0,
        "top3_TN_0.4": 89,
        "top3_FN_0.4": 0,
        "top3_accuracy_0.5": 1.0,
        "top3_precision_0.5": 1.0,
        "top3_recall_0.5": 1.0,
        "top3_f1_score_0.5": 1.0,
        "top3_TP_0.5": 1215,
        "top3_FP_0.5": 0,
        "top3_TN_0.5": 89,
        "top3_FN_0.5": 0,
        "top3_accuracy_0.6": 1.0,
        "top3_precision_0.6": 1.0,
        "top3_recall_0.6": 1.0,
        "top3_f1_score_0.6": 1.0,
        "top3_TP_0.6": 1215,
        "top3_FP_0.6": 0,
        "top3_TN_0.6": 89,
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
        "top3_acc_novel_only": 0.39642,
        "top1_acc_novel_only": 0.31942,
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
        "M_nrp_post_top3": 43.19571115687621,
        "M_nrp_post_top1": 39.70959360509205,
    }
