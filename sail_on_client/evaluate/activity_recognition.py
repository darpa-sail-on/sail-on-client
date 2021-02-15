"""Activity Recognition Class for metrics for sail-on."""

from sail_on_client.evaluate.metrics import ProgramMetrics
from evaluate.metrics import M_acc, M_num, M_ndp, M_num_stats
from evaluate.metrics import M_ndp_failed_reaction
from evaluate.metrics import M_accuracy_on_novel

import numpy as np

from typing import Dict


class ActivityRecognitionMetrics(ProgramMetrics):
    """Activity Recognition program metric class."""

    def __init__(
        self,
        protocol: str,
        video_id: int,
        novel: int,
        detection: int,
        classification: int,
        spatial: int,
        temporal: int,
    ) -> None:
        """Initialize."""
        super().__init__(protocol)
        self.activity_id = id
        self.novel_id = novel
        self.detection_id = detection
        self.classification_id = classification
        self.spatial_id = spatial
        self.temporal_id = temporal

    def m_acc(
        self,
        gt_novel: np.ndarray,
        p_class: np.ndarray,
        gt_class: np.ndarray,
        round_size: int,
        asymptotic_start_round: int,
    ) -> Dict:
        """m_acc function."""
        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return M_acc(
            gt_novel, class_prob, gt_class_idx, round_size, asymptotic_start_round
        )

    def m_num(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> float:
        """m_num function."""
        return M_num(p_novel, gt_novel)

    def m_num_stats(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_num_stats function."""
        return M_num_stats(p_novel, gt_novel)

    def m_ndp(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_ndp function."""
        return M_ndp(p_novel, gt_novel)

    def m_ndp_pre(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_ndp_pre function."""
        return M_ndp(p_novel, gt_novel, mode="pre_novelty")

    def m_ndp_post(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """m_ndp_post function."""
        return M_ndp(p_novel, gt_novel, mode="post_novelty")

    def m_ndp_failed_reaction(
        self,
        p_novel: np.ndarray,
        gt_novel: np.ndarray,
        p_class: np.ndarray,
        gt_class: np.ndarray,
    ) -> Dict:
        """m_ndp_failed_reaction function."""
        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return M_ndp_failed_reaction(p_novel, gt_novel, class_prob, gt_class_idx)

    def m_accuracy_on_novel(
        self, p_class: np.ndarray, gt_class: np.ndarray, gt_novel: np.ndarray
    ) -> Dict:
        """m_accuracy_on_novel function."""
        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return M_accuracy_on_novel(class_prob, gt_class_idx, gt_novel)

    def m_is_cdt_and_is_early(self, gt_idx: int, ta2_idx: int, test_len: int) -> Dict:
        """Is CDT and is early."""
        is_cdt = (ta2_idx >= gt_idx) & (ta2_idx < test_len)
        is_early = ta2_idx < gt_idx
        return {"Is CDT": is_cdt, "Is Early": is_early}
