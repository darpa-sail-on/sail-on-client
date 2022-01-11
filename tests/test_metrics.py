"""Tests for metric functions."""

from sail_on_client.evaluate.metrics import (
    m_num,
    m_num_stats,
    m_ndp,
    m_ndp_pre,
    m_ndp_post,
    m_ndp_failed_reaction,
    m_acc,
)

import pytest
import os
import pandas as pd


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


def test_m_num(detection_files, expected_ar_m_num_values):
    """
    Test m_num computation.

    Args:
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_num_values:  Values expected_ar for m_num

    Return:
        None
    """
    detection, gt = detection_files
    m_num_val = m_num(detection[1], gt[1])
    assert m_num_val == expected_ar_m_num_values


def test_m_num_stats(detection_files, expected_ar_m_num_stats_values):
    """
    Test m_num_stats computation.

    Args:
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_num_stats_values:  Values expected_ar for m_num_stats

    Return:
        None
    """
    detection, gt = detection_files
    m_num_stats_val = m_num_stats(detection[1], gt[1])
    assert m_num_stats_val == expected_ar_m_num_stats_values


def test_m_ndp(detection_files, expected_ar_m_ndp_values):
    """
    Test m_ndp computation.

    Args:
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_ndp_values: Values expected_ar for m_ndp

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_val = m_ndp(detection[1], gt[1])
    assert m_ndp_val == expected_ar_m_ndp_values


def test_m_ndp_pre(detection_files, expected_ar_m_ndp_pre_values):
    """
    Test m_ndp_pre computation.

    Args:
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_ndp_pre_values: Values expected_ar from m_ndp_pre

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_pre_val = m_ndp_pre(detection[1], gt[1])
    assert m_ndp_pre_val == expected_ar_m_ndp_pre_values


def test_m_ndp_post(detection_files, expected_ar_m_ndp_post_values):
    """
    Test m_ndp_post computation.

    Args:
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        expected_ar_m_ndp_post_values: Values expected_ar for m_ndp_post

    Return:
        None
    """
    detection, gt = detection_files
    m_ndp_post_val = m_ndp_post(detection[1], gt[1])
    assert m_ndp_post_val == expected_ar_m_ndp_post_values


def test_m_ndp_failed_reaction(
    detection_files, classification_file, expected_ar_m_ndp_failed_values
):
    """
    Test m_ndp_failed_reaction computation.

    Args:
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_ar_m_ndp_failed_values: Values expected_ar for m_ndp_failed_values

    Return:
        None
    """
    detection, gt = detection_files
    class_prob = classification_file.iloc[
        :, range(1, classification_file.shape[1])
    ].to_numpy()
    gt_class_idx = gt[3].to_numpy()
    m_ndp_failed = m_ndp_failed_reaction(detection[1], gt[1], class_prob, gt_class_idx)
    assert m_ndp_failed == expected_ar_m_ndp_failed_values


def test_m_acc(detection_files, classification_file, expected_ar_m_acc_values):
    """
    Test m_acc computation.

    Args:
        detection_files (Tuple): A tuple of data frames containing detection and ground truth
        classification_file: A data frame containing classification
        expected_ar_m_acc_values: Values expected_ar for m_acc

    Return:
        None
    """
    detection, gt = detection_files
    class_prob = classification_file.iloc[
        :, range(1, classification_file.shape[1])
    ].to_numpy()
    gt_class_idx = gt[3].to_numpy()
    m_acc_val = m_acc(gt[1], class_prob, gt_class_idx, 100, 5)
    assert m_acc_val == expected_ar_m_acc_values
