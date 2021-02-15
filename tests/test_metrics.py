"""Tests for base metric class."""

from sail_on_client.evaluate.metrics import ProgramMetrics
import numpy as np
import pytest


@pytest.fixture(scope="function")
def program_metrics():
    """
    Fixture for initializing program metric.

    Return:
        None
    """
    program_metrics = ProgramMetrics("OND")
    return program_metrics


@pytest.mark.parametrize("protocol_name", ["OND", "CONDDA"])
def test_initialize(protocol_name):
    """
    Test program metric initialization.

    Return:
        None
    """
    program_metrics = ProgramMetrics(protocol_name)
    assert program_metrics.protocol == protocol_name


def test_m_acc(program_metrics):
    """
    Test m_acc computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_acc(
            np.random.random((10, 1)),
            np.random.random((10, 1)),
            np.random.random((10, 3)),
            10,
            10,
        )


def test_m_num(program_metrics):
    """
    Test m_num computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_num(np.random.random((10, 1)), np.random.random((10, 1)))


def test_m_num_stats(program_metrics):
    """
    Test m_num_stats computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_num_stats(
            np.random.random((10, 1)), np.random.random((10, 1))
        )


def test_m_ndp(program_metrics):
    """
    Test m_ndp computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_ndp(np.random.random((10, 1)), np.random.random((10, 1)))


def test_m_ndp_pre(program_metrics):
    """
    Test m_ndp_pre computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_ndp_pre(np.random.random((10, 1)), np.random.random((10, 1)))


def test_m_ndp_post(program_metrics):
    """
    Test m_ndp_post computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_ndp_post(np.random.random((10, 1)), np.random.random((10, 1)))


def test_m_ndp_failed_reaction(program_metrics):
    """
    Test m_ndp_failed_reaction computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_ndp_failed_reaction(
            np.random.random((10, 1)),
            np.random.random((10, 1)),
            np.random.random((10, 3)),
            np.random.random((10, 3)),
        )


def test_m_accuracy_on_novel(program_metrics):
    """
    Test m_accuracy_on_novel computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_accuracy_on_novel(
            np.random.random((10, 1)),
            np.random.random((10, 1)),
            np.random.random((10, 3)),
        )


def test_is_cdt_and_is_early(program_metrics):
    """
    Test m_is_cdt_and_is_early computation.

    Args:
        program_metrics (ProgramMetrics): An instance of ProgramMetrics

    Return:
        None
    """
    with pytest.raises(NotImplementedError):
        program_metrics.m_is_cdt_and_is_early(10, 20, 30)
