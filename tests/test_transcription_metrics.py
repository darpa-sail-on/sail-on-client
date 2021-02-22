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
    detection_file = os.path.join(result_folder, "OND.0.90001.8714062_detection.csv")
    detection = pd.read_csv(detection_file, sep=",", header=None)
    return detection, gt


@pytest.fixture(scope="function")
def classification_file():
    """Fixture for reading classification file."""
    result_folder = os.path.join(
        os.path.dirname(__file__), "mock_results", "transcripts"
    )
    classification_file_id = os.path.join(
        result_folder, "OND.0.90001.8714062_classification.csv"
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
    assert m_num == 1


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
    assert m_num_stats["GT_indx"] == 270 and m_num_stats["P_indx"] == 5


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
        "FN": 38,
        "FP": 53,
        "TN": 338,
        "TP": 83,
        "accuracy": 0.82227,
        "f1_score": 0.64591,
        "precision": 0.61029,
        "recall": 0.68595,
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
        "FN": 0,
        "FP": 36,
        "TN": 233,
        "TP": 0,
        "accuracy": 0.86617,
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
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
        "FN": 38,
        "FP": 17,
        "TN": 105,
        "TP": 83,
        "accuracy": 0.77366,
        "f1_score": 0.75113,
        "precision": 0.83,
        "recall": 0.68595,
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
        "top1_FN": 28,
        "top1_FP": 50,
        "top1_TN": 40,
        "top1_TP": 0,
        "top1_accuracy": 0.33898,
        "top1_f1_score": 0.0,
        "top1_precision": 0.0,
        "top1_recall": 0.0,
        "top3_FN": 22,
        "top3_FP": 15,
        "top3_TN": 8,
        "top3_TP": 0,
        "top3_accuracy": 0.17778,
        "top3_f1_score": 0.0,
        "top3_precision": 0.0,
        "top3_recall": 0.0,
    }


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
        m_num_stats["GT_indx"], m_num_stats["P_indx"], gt.shape[0]
    )
    assert not is_cdt_is_early["Is CDT"] and is_cdt_is_early["Is Early"]
