"""Abstract Class for metrics for sail-on."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


class ProgramMetrics(ABC):
    """Abstract program metric class."""

    def __init__(self, protocol: str) -> None:
        """
        Initialize.

        Args:
            protocol: Name of the protocol.

        Returns:
            None
        """
        self.protocol = protocol

    @abstractmethod
    def m_acc(
        self,
        gt_novel: np.ndarray,
        p_class: np.ndarray,
        gt_class: np.ndarray,
        round_size: int,
        asymptotic_start_round: int,
    ) -> Dict:
        """
        m_acc abstract function.

        Args:
            gt_novel: ground truth detections
            p_class: class predictions
            gt_class: ground truth classes
            round_size: size of the round
            asymptotic_start_round: asymptotic samples considered for computing metrics

        Returns:
            Dictionary containing top1, top3 accuracy over the test, pre and post novelty.
        """

    @abstractmethod
    def m_acc_round_wise(
        self, p_class: np.ndarray, gt_class: np.ndarray, round_id: int
    ) -> Dict:
        """
        m_acc_round_wise abstract function.

        Args:
            p_class: class predictions
            gt_class: ground truth classes

        Returns:
            Dictionary containing top1, top3 accuracy for a round
        """

    @abstractmethod
    def m_num(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_num abstract function.

        Args:
            p_novel: detection predictions
            gt_novel: ground truth detections

        Returns:
            Difference between the novelty introduction and predicting change in world.
        """

    @abstractmethod
    def m_num_stats(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_num_stats abstract function.

        Args:
            p_novel: detection predictions
            gt_novel: ground truth detections

        Returns:
            Dictionary containing indices for novelty introduction and change in world prediction.
        """

    @abstractmethod
    def m_ndp(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp abstract function.

        Args:
            p_novel: detection predictions
            gt_novel: ground truth detections

        Returns:
            Dictionary containing novelty detection performance over the test.
        """

    @abstractmethod
    def m_ndp_pre(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp_pre abstract function.

        Args:
            p_novel: detection predictions
            gt_novel: ground truth detections

        Returns:
            Dictionary containing detection performance pre novelty.
        """

    @abstractmethod
    def m_ndp_post(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp_post abstract function.

        Args:
            p_novel: detection predictions
            gt_novel: ground truth detections

        Returns:
            Dictionary containing detection performance post novelty.
        """

    @abstractmethod
    def m_ndp_failed_reaction(
        self,
        p_novel: np.ndarray,
        gt_novel: np.ndarray,
        p_class: np.ndarray,
        gt_class: np.ndarray,
    ) -> Dict:
        """
        m_ndp_failed_reaction abstract function.

        Args:
            p_novel: detection predictions
            gt_novel: ground truth detections
            p_class: class predictions
            gt_class: ground truth classes

        Returns:
            Dictionary containing TP, FP, TN, FN, top1, top3 accuracy over the test.
        """

    @abstractmethod
    def m_accuracy_on_novel(
        self, p_novel: np.ndarray, gt_class: np.ndarray, gt_novel: np.ndarray
    ) -> Dict:
        """
        m_accuracy_on_novel abstract function.

        Args:
            p_novel: detection predictions
            gt_class: ground truth classes
            gt_novel: ground truth detections

        Returns:
            Accuracy on novely samples
        """

    @abstractmethod
    def m_is_cdt_and_is_early(self, gt_idx: int, ta2_idx: int, test_len: int) -> Dict:
        """
        m_is_cdt_and_is_early abstract function.

        Args:
            gt_idx: Index when novelty is introduced
            ta2_idx: Index when change is detected
            test_len: Length of test

        Returns
            Dictionary containing boolean showing if change was was detected and if it was detected early
        """
