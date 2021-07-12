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


def test_m_acc(dtm_metrics, detection_files, classification_file, expected_dt_m_acc_values):
    """
    Test m_acc computation.

    Args:
        program_metrics (DocumentTranscriptionMetrics): An instance of DocumentTranscriptionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_dt_m_acc_values: Expected values of m_acc

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
    assert m_acc == expected_dt_m_acc_values


def test_m_acc_round_wise(dtm_metrics, detection_files, classification_file,
                          expected_dt_m_acc_roundwise_values):
    """
    Test m_acc computation for a round.

    Args:
        dtm_metrics (DocumentTranscriptionMetrics): An instance of DocumentTranscriptionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_dt_m_acc_roundwise_values: Expected values of m_acc for a round

    Return:
        None
    """
    _, gt = detection_files
    m_acc_round_wise = dtm_metrics.m_acc_round_wise(
        classification_file, gt[dtm_metrics.classification_id], 0
    )
    assert m_acc_round_wise == expected_dt_m_acc_roundwise_values


def test_m_num(dtm_metrics, detection_files, expected_dt_m_num_values):
    """
    Test m_num computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_dt_m_num_values: Expected values of m_num

    Return:
        None
    """
    detection, gt = detection_files
    m_num = dtm_metrics.m_num(detection[1], gt[dtm_metrics.novel_id])
    assert m_num == expected_dt_m_num_values


def test_m_num_stats(dtm_metrics, detection_files, expected_dt_m_num_stats_values):
    """
    Test m_num_stats computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_dt_m_num_stats_values: Expected values of m_num_stats

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats = dtm_metrics.m_num_stats(detection[1], gt[dtm_metrics.novel_id])
    assert m_num_stats == expected_dt_m_num_stats_values


def test_m_ndp(dtm_metrics, detection_files, expected_dt_m_ndp_values):
    """
    Test m_ndp computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_dt_m_ndp_values: Expected values of m_ndp

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp = dtm_metrics.m_ndp(detection[1], gt[dtm_metrics.novel_id])
    assert m_ndp == expected_dt_m_ndp_values


def test_m_ndp_pre(dtm_metrics, detection_files, expected_dt_m_ndp_pre_values):
    """
    Test m_ndp_pre computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_dt_m_ndp_pre_values: Expected m_ndp_pre values

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_pre = dtm_metrics.m_ndp_pre(detection[1], gt[dtm_metrics.novel_id])
    assert m_ndp_pre == expected_dt_m_ndp_pre_values


def test_m_ndp_post(dtm_metrics, detection_files, expected_dt_m_ndp_post_values):
    """
    Test m_ndp_post computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_dt_m_ndp_post_values: Expected ndp post values

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_post = dtm_metrics.m_ndp_post(detection[1], gt[dtm_metrics.novel_id])
    assert m_ndp_post == expected_dt_m_ndp_post_values

def test_m_ndp_failed_reaction(dtm_metrics, detection_files, classification_file,
        expected_dt_m_ndp_failed_values):
    """
    Test m_ndp_failed_reaction computation.

    Args:
        dtm_metrics (DocumentTranscription): An instance of DocumentTranscription
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_dt_m_ndp_failed_values: Expected ndp failed values

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
    assert m_ndp_failed == expected_dt_m_ndp_failed_values


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
    dtm_metrics, detection_files, classification_file, baseline_classification_file,
    expected_dt_m_nrp_values
):
    """
    Test novelty reaction performance.

    Args:
        dtm_metrics (DocumentTranscriptionMetrics): An instance of DocumentTranscriptionMetrics
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file (DataFrame): A data frame with classification prediction
        baseline_classification_fDataFrame): A data frame with classification prediction for baseline
        expected_dt_m_nrp_values: Expected values of nrp

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
    assert m_nrp == expected_dt_m_nrp_values
