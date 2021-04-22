"""Tests for activity recognition metric class."""

from sail_on_client.evaluate.document_transcription import DocumentTranscriptionMetrics

import os
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def dtm_metrics():
    """Fixture for generated document transcription metrics."""
    gt_config = list(range(0, 11))
    dtm_metrics = DocumentTranscriptionMetrics("OND", *gt_config)
    return dtm_metrics


@pytest.fixture(scope="function")
def detection_files():
    """Fixture for reading detection file and ground truth."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    gt_file = os.path.join(result_folder, "OND.0.90001.8714062_single_df.csv")
    gt = pd.read_csv(gt_file, sep=",", header=None, skiprows=1)
    detection_file = os.path.join(
        result_folder, "OND.0.90001.8714062_PreComputedDetector_detection.csv"
    )
    detection = pd.read_csv(detection_file, sep=",", header=None)
    return detection, gt


@pytest.fixture(scope="function")
def classification_file():
    """Fixture for reading classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.0.90001.8714062_PreComputedDetector_classification.csv"
    )
    classification = pd.read_csv(classification_file_id, sep=",", header=None)
    return classification


@pytest.fixture(scope="function")
def baseline_classification_file():
    """Fixture for reading baseline classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    classification_file_id = os.path.join(
        result_folder,
        "OND.0.90001.8714062_BaselinePreComputedDetector_classification.csv",
    )
    classification = pd.read_csv(classification_file_id, sep=",", header=None)
    return classification


@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_initialize(protocol_name):
    """
    Test document transcription metric initialization.

    Return:
        None
    """
    gt_config = list(range(0, 11))
    dtm_metrics = DocumentTranscriptionMetrics(protocol_name, *gt_config)
    assert dtm_metrics.protocol == protocol_name


def test_m_acc(dtm_metrics, detection_files, classification_file):
    """
    Test m_acc computation.

    Args:
        program_metrics (DocumentTranscriptionMetrics): An instance of DocumentTranscriptionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    m_acc = dtm_metrics.m_acc(
        gt[dtm_metrics.novel_id],
        classification_file,
        gt[dtm_metrics.classification_id],
        100,
        5,
    )
    assert m_acc == {
        "asymptotic_500_top1": 0.766,
        "asymptotic_500_top3": 0.91,
        "full_top1": 0.76953,
        "full_top3": 0.91211,
        "post_top1": 0.76543,
        "post_top3": 0.86831,
        "pre_top1": 0.77323,
        "pre_top3": 0.95167,
    }


def test_m_acc_round_wise(dtm_metrics, detection_files, classification_file):
    """
    Test m_acc computation for a round.

    Args:
        dtm_metrics (DocumentTranscriptionMetrics): An instance of DocumentTranscriptionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    _, gt = detection_files
    m_acc_round_wise = dtm_metrics.m_acc_round_wise(
        classification_file, gt[dtm_metrics.classification_id], 0
    )
    assert m_acc_round_wise == {
        "top1_accuracy_round_0": 0.76953,
        "top3_accuracy_round_0": 0.91211,
    }


def test_m_num(dtm_metrics, detection_files):
    """
    Test m_num computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num = dtm_metrics.m_num(detection[1], gt[dtm_metrics.novel_id])
    assert m_num == {"0.175": 1, "0.225": 1, "0.3": 1, "0.4": 1, "0.5": 1, "0.6": 1}


def test_m_num_stats(dtm_metrics, detection_files):
    """
    Test m_num_stats computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = dtm_metrics.m_num_stats(detection[1], gt[dtm_metrics.novel_id])
    assert m_num_stats == {
        "GT_indx": 270,
        "P_indx_0.175": 1,
        "P_indx_0.225": 1,
        "P_indx_0.3": 1,
        "P_indx_0.4": 5,
        "P_indx_0.5": 5,
        "P_indx_0.6": 5,
    }


def test_m_ndp(dtm_metrics, detection_files):
    """
    Test m_ndp computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp = dtm_metrics.m_ndp(detection[1], gt[dtm_metrics.novel_id])
    assert m_ndp == {
        "accuracy_0.175": 0.63086,
        "precision_0.175": 0.38889,
        "recall_0.175": 0.98347,
        "f1_score_0.175": 0.55738,
        "TP_0.175": 119,
        "FP_0.175": 187,
        "TN_0.175": 204,
        "FN_0.175": 2,
        "accuracy_0.225": 0.64844,
        "precision_0.225": 0.4,
        "recall_0.225": 0.97521,
        "f1_score_0.225": 0.56731,
        "TP_0.225": 118,
        "FP_0.225": 177,
        "TN_0.225": 214,
        "FN_0.225": 3,
        "accuracy_0.3": 0.69141,
        "precision_0.3": 0.43123,
        "recall_0.3": 0.95868,
        "f1_score_0.3": 0.59487,
        "TP_0.3": 116,
        "FP_0.3": 153,
        "TN_0.3": 238,
        "FN_0.3": 5,
        "accuracy_0.4": 0.74805,
        "precision_0.4": 0.48131,
        "recall_0.4": 0.85124,
        "f1_score_0.4": 0.61493,
        "TP_0.4": 103,
        "FP_0.4": 111,
        "TN_0.4": 280,
        "FN_0.4": 18,
        "accuracy_0.5": 0.82227,
        "precision_0.5": 0.61029,
        "recall_0.5": 0.68595,
        "f1_score_0.5": 0.64591,
        "TP_0.5": 83,
        "FP_0.5": 53,
        "TN_0.5": 338,
        "FN_0.5": 38,
        "accuracy_0.6": 0.83008,
        "precision_0.6": 0.66667,
        "recall_0.6": 0.56198,
        "f1_score_0.6": 0.60987,
        "TP_0.6": 68,
        "FP_0.6": 34,
        "TN_0.6": 357,
        "FN_0.6": 53,
    }


def test_m_ndp_pre(dtm_metrics, detection_files):
    """
    Test m_ndp_pre computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_pre = dtm_metrics.m_ndp_pre(detection[1], gt[dtm_metrics.novel_id])
    assert m_ndp_pre == {
        "accuracy_0.175": 0.5316,
        "precision_0.175": 0.0,
        "recall_0.175": 0.0,
        "f1_score_0.175": 0.0,
        "TP_0.175": 0,
        "FP_0.175": 126,
        "TN_0.175": 143,
        "FN_0.175": 0,
        "accuracy_0.225": 0.56134,
        "precision_0.225": 0.0,
        "recall_0.225": 0.0,
        "f1_score_0.225": 0.0,
        "TP_0.225": 0,
        "FP_0.225": 118,
        "TN_0.225": 151,
        "FN_0.225": 0,
        "accuracy_0.3": 0.62825,
        "precision_0.3": 0.0,
        "recall_0.3": 0.0,
        "f1_score_0.3": 0.0,
        "TP_0.3": 0,
        "FP_0.3": 100,
        "TN_0.3": 169,
        "FN_0.3": 0,
        "accuracy_0.4": 0.73978,
        "precision_0.4": 0.0,
        "recall_0.4": 0.0,
        "f1_score_0.4": 0.0,
        "TP_0.4": 0,
        "FP_0.4": 70,
        "TN_0.4": 199,
        "FN_0.4": 0,
        "accuracy_0.5": 0.86617,
        "precision_0.5": 0.0,
        "recall_0.5": 0.0,
        "f1_score_0.5": 0.0,
        "TP_0.5": 0,
        "FP_0.5": 36,
        "TN_0.5": 233,
        "FN_0.5": 0,
        "accuracy_0.6": 0.9145,
        "precision_0.6": 0.0,
        "recall_0.6": 0.0,
        "f1_score_0.6": 0.0,
        "TP_0.6": 0,
        "FP_0.6": 23,
        "TN_0.6": 246,
        "FN_0.6": 0,
    }


def test_m_ndp_post(dtm_metrics, detection_files):
    """
    Test m_ndp_post computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_post = dtm_metrics.m_ndp_post(detection[1], gt[dtm_metrics.novel_id])
    assert m_ndp_post == {
        "accuracy_0.175": 0.74074,
        "precision_0.175": 0.66111,
        "recall_0.175": 0.98347,
        "f1_score_0.175": 0.7907,
        "TP_0.175": 119,
        "FP_0.175": 61,
        "TN_0.175": 61,
        "FN_0.175": 2,
        "accuracy_0.225": 0.74486,
        "precision_0.225": 0.66667,
        "recall_0.225": 0.97521,
        "f1_score_0.225": 0.79195,
        "TP_0.225": 118,
        "FP_0.225": 59,
        "TN_0.225": 63,
        "FN_0.225": 3,
        "accuracy_0.3": 0.76132,
        "precision_0.3": 0.68639,
        "recall_0.3": 0.95868,
        "f1_score_0.3": 0.8,
        "TP_0.3": 116,
        "FP_0.3": 53,
        "TN_0.3": 69,
        "FN_0.3": 5,
        "accuracy_0.4": 0.7572,
        "precision_0.4": 0.71528,
        "recall_0.4": 0.85124,
        "f1_score_0.4": 0.77736,
        "TP_0.4": 103,
        "FP_0.4": 41,
        "TN_0.4": 81,
        "FN_0.4": 18,
        "accuracy_0.5": 0.77366,
        "precision_0.5": 0.83,
        "recall_0.5": 0.68595,
        "f1_score_0.5": 0.75113,
        "TP_0.5": 83,
        "FP_0.5": 17,
        "TN_0.5": 105,
        "FN_0.5": 38,
        "accuracy_0.6": 0.73663,
        "precision_0.6": 0.86076,
        "recall_0.6": 0.56198,
        "f1_score_0.6": 0.68,
        "TP_0.6": 68,
        "FP_0.6": 11,
        "TN_0.6": 111,
        "FN_0.6": 53,
    }


def test_m_ndp_failed_reaction(dtm_metrics, detection_files, classification_file):
    """
    Test m_ndp_failed_reaction computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_failed = dtm_metrics.m_ndp_failed_reaction(
        detection[1],
        gt[dtm_metrics.novel_id],
        classification_file,
        gt[dtm_metrics.classification_id],
    )
    assert m_ndp_failed == {
        "top1_accuracy_0.175": 0.22034,
        "top1_precision_0.175": 0.22414,
        "top1_recall_0.175": 0.92857,
        "top1_f1_score_0.175": 0.36111,
        "top1_TP_0.175": 26,
        "top1_FP_0.175": 90,
        "top1_TN_0.175": 0,
        "top1_FN_0.175": 2,
        "top1_accuracy_0.225": 0.21186,
        "top1_precision_0.225": 0.21739,
        "top1_recall_0.225": 0.89286,
        "top1_f1_score_0.225": 0.34965,
        "top1_TP_0.225": 25,
        "top1_FP_0.225": 90,
        "top1_TN_0.225": 0,
        "top1_FN_0.225": 3,
        "top1_accuracy_0.3": 0.19492,
        "top1_precision_0.3": 0.20354,
        "top1_recall_0.3": 0.82143,
        "top1_f1_score_0.3": 0.32624,
        "top1_TP_0.3": 23,
        "top1_FP_0.3": 90,
        "top1_TN_0.3": 0,
        "top1_FN_0.3": 5,
        "top1_accuracy_0.4": 0.18644,
        "top1_precision_0.4": 0.12222,
        "top1_recall_0.4": 0.39286,
        "top1_f1_score_0.4": 0.18644,
        "top1_TP_0.4": 11,
        "top1_FP_0.4": 79,
        "top1_TN_0.4": 11,
        "top1_FN_0.4": 17,
        "top1_accuracy_0.5": 0.33898,
        "top1_precision_0.5": 0.0,
        "top1_recall_0.5": 0.0,
        "top1_f1_score_0.5": 0.0,
        "top1_TP_0.5": 0,
        "top1_FP_0.5": 50,
        "top1_TN_0.5": 40,
        "top1_FN_0.5": 28,
        "top1_accuracy_0.6": 0.47458,
        "top1_precision_0.6": 0.0,
        "top1_recall_0.6": 0.0,
        "top1_f1_score_0.6": 0.0,
        "top1_TP_0.6": 0,
        "top1_FP_0.6": 34,
        "top1_TN_0.6": 56,
        "top1_FN_0.6": 28,
        "top3_accuracy_0.175": 0.44444,
        "top3_precision_0.175": 0.46512,
        "top3_recall_0.175": 0.90909,
        "top3_f1_score_0.175": 0.61538,
        "top3_TP_0.175": 20,
        "top3_FP_0.175": 23,
        "top3_TN_0.175": 0,
        "top3_FN_0.175": 2,
        "top3_accuracy_0.225": 0.42222,
        "top3_precision_0.225": 0.45238,
        "top3_recall_0.225": 0.86364,
        "top3_f1_score_0.225": 0.59375,
        "top3_TP_0.225": 19,
        "top3_FP_0.225": 23,
        "top3_TN_0.225": 0,
        "top3_FN_0.225": 3,
        "top3_accuracy_0.3": 0.37778,
        "top3_precision_0.3": 0.425,
        "top3_recall_0.3": 0.77273,
        "top3_f1_score_0.3": 0.54839,
        "top3_TP_0.3": 17,
        "top3_FP_0.3": 23,
        "top3_TN_0.3": 0,
        "top3_FN_0.3": 5,
        "top3_accuracy_0.4": 0.17778,
        "top3_precision_0.4": 0.22222,
        "top3_recall_0.4": 0.27273,
        "top3_f1_score_0.4": 0.2449,
        "top3_TP_0.4": 6,
        "top3_FP_0.4": 21,
        "top3_TN_0.4": 2,
        "top3_FN_0.4": 16,
        "top3_accuracy_0.5": 0.17778,
        "top3_precision_0.5": 0.0,
        "top3_recall_0.5": 0.0,
        "top3_f1_score_0.5": 0.0,
        "top3_TP_0.5": 0,
        "top3_FP_0.5": 15,
        "top3_TN_0.5": 8,
        "top3_FN_0.5": 22,
        "top3_accuracy_0.6": 0.24444,
        "top3_precision_0.6": 0.0,
        "top3_recall_0.6": 0.0,
        "top3_f1_score_0.6": 0.0,
        "top3_TP_0.6": 0,
        "top3_FP_0.6": 12,
        "top3_TN_0.6": 11,
        "top3_FN_0.6": 22,
    }


@pytest.mark.skip(reason="no way of currently testing this")
def test_m_accuracy_on_novel(dtm_metrics, detection_files, classification_file):
    """
    Test m_accuracy_on_novel computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification

    Return:
        None
    """
    detection, gt = detection_files
    dtm_metrics.m_accuracy_on_novel(
        classification_file, gt[dtm_metrics.classification_id], gt[dtm_metrics.novel_id]
    )


def test_is_cdt_and_is_early(dtm_metrics, detection_files):
    """
    Test m_is_cdt_and_is_early computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = dtm_metrics.m_num_stats(detection[1], gt[dtm_metrics.novel_id])
    is_cdt_is_early = dtm_metrics.m_is_cdt_and_is_early(
        m_num_stats["GT_indx"], m_num_stats["P_indx_0.5"], gt.shape[0]
    )
    assert not is_cdt_is_early["Is CDT"] and is_cdt_is_early["Is Early"]


def test_m_nrp(
    dtm_metrics, detection_files, classification_file, baseline_classification_file
):
    """
    Test novelty reaction performance.

    Args:
        dtm_metrics (DocumentTranscriptionMetrics): An instance of DocumentTranscriptionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file (DataFrame): A data frame with classification prediction
        baseline_classification_fDataFrame): A data frame with classification prediction for baseline

    Return:
        None
    """
    detection, gt = detection_files
    m_acc = dtm_metrics.m_acc(
        gt[dtm_metrics.novel_id],
        classification_file,
        gt[dtm_metrics.classification_id],
        100,
        5,
    )
    m_acc_baseline = dtm_metrics.m_acc(
        gt[dtm_metrics.detection_id],
        baseline_classification_file,
        gt[dtm_metrics.classification_id],
        100,
        5,
    )
    m_nrp = dtm_metrics.m_nrp(m_acc, m_acc_baseline)
    assert m_nrp == {
        "M_nrp_post_top3": 86.831,
        "M_nrp_post_top1": 76.543,
    }
