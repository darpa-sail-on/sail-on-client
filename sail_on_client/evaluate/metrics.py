"""Abstract Class for metrics for sail-on."""

import numpy as np
from typing import Dict


class ProgramMetrics:
    """Abstract program metric class."""

    def __init__(self, protocol: str) -> None:
        """Initialize."""
        self.protocol = protocol

    def m_acc(
        self,
        gt_novel: np.ndarray,
        p_class: np.ndarray,
        gt_class: np.ndarray,
        round_size: int,
        asymptotic_start_round: int,
    ) -> Dict:
        """m_acc abstract function."""
        raise NotImplementedError("Calling m_acc for an abstract class")

    def m_num(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> float:
        """m_num abstract function."""
        raise NotImplementedError("Calling m_num for an abstract class")

    def m_num_stats(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_num_stats abstract function."""
        raise NotImplementedError("Calling m_num_stats for an abstract class")

    def m_ndp(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_ndp abstract function."""
        raise NotImplementedError("Calling m_ndp for an abstract class")

    def m_ndp_pre(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_ndp_pre abstract function."""
        raise NotImplementedError("Calling m_ndp_pre for an abstract class")

    def m_ndp_post(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_ndp_post abstract function."""
        raise NotImplementedError("Calling m_ndp_post for an abstract class")

    def m_ndp_failed_reaction(
        self,
        p_novel: np.ndarray,
        gt_novel: np.ndarray,
        p_class: np.ndarray,
        gt_class: np.ndarray,
    ) -> Dict:
        """m_ndp_failed_reaction abstract function."""
        raise NotImplementedError("Calling m_ndp_failed_reaction for an abstract class")

    def m_accuracy_on_novel(
        self, p_novel: np.ndarray, gt_class: np.ndarray, gt_novel: np.ndarray
    ) -> Dict:
        """m_accuracy_on_novel abstract function."""
        raise NotImplementedError("Calling m_accuracy_on_novel for an abstract class")

    def m_is_cdt_and_is_early(self, gt_idx: int, ta2_idx: int, test_len: int) -> Dict:
        """m_is_cdt_and_is_early abstract function."""
        raise NotImplementedError("Calling m_is_cdt_and_is_early for an abstract class")
