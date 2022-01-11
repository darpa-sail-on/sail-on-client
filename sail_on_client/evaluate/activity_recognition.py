"""Activity Recognition Class for metrics for sail-on."""

from sail_on_client.evaluate.program_metrics import ProgramMetrics
from sail_on_client.evaluate.metrics import m_acc, m_num, m_ndp, m_num_stats
from sail_on_client.evaluate.metrics import m_ndp_failed_reaction
from sail_on_client.evaluate.metrics import m_accuracy_on_novel
from sail_on_client.evaluate.utils import topk_accuracy

import numpy as np
from pandas import DataFrame

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
        """
        Initialize.

        Args:
            protocol: Name of the protocol.
            video_id: Column id for video
            novel: Column id for predicting if change was detected
            detection: Column id for predicting sample wise novelty
            classification: Column id for predicting sample wise classes
            spatial: Column id for predicting spatial attribute
            temporal: Column id for predicting temporal attribute

        Returns:
            None
        """
        super().__init__(protocol)
        self.activity_id = video_id
        self.novel_id = novel
        self.detection_id = detection
        self.classification_id = classification
        self.spatial_id = spatial
        self.temporal_id = temporal

    def m_acc(
        self,
        gt_novel: DataFrame,
        p_class: DataFrame,
        gt_class: DataFrame,
        round_size: int,
        asymptotic_start_round: int,
    ) -> Dict:
        """
        m_acc function.

        Args:
            gt_novel: ground truth detections for N videos (Dimension: N X 1)
            p_class: class predictions with video id for N videos (Dimension: N X 90 [vid,novel_class,88 known class])
            gt_class: ground truth classes for N videos (Dimension: N X 1)
            round_size: size of the round
            asymptotic_start_round: asymptotic samples considered for computing metrics

        Returns:
            Dictionary containing top1, top3 accuracy over the test, pre and post novelty.
        """
        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return m_acc(
            gt_novel, class_prob, gt_class_idx, round_size, asymptotic_start_round
        )

    def m_acc_round_wise(
        self, p_class: DataFrame, gt_class: DataFrame, round_id: int
    ) -> Dict:
        """
        m_acc_round_wise function.

        Args:
            p_class: detection predictions
            gt_class: ground truth classes
            round_id: round identifier

        Returns:
            Dictionary containing top1, top3 accuracy for a round
        """
        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        top1_acc = topk_accuracy(class_prob, gt_class_idx, k=1)
        top3_acc = topk_accuracy(class_prob, gt_class_idx, k=3)
        return {
            f"top1_accuracy_round_{round_id}": top1_acc,
            f"top3_accuracy_round_{round_id}": top3_acc,
        }

    def m_num(self, p_novel: DataFrame, gt_novel: DataFrame) -> Dict:
        """
        m_num function.

        Args:
            p_novel: detection predictions for N videos (Dimension: N X 1)
            gt_novel: ground truth detections for N videos (Dimension: N X 1)

        Returns:
            Difference between the novelty introduction and predicting change in world.
        """
        return m_num(p_novel, gt_novel)

    def m_num_stats(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_num_stats function.

        Args:
            p_novel: detection predictions for N videos (Dimension: N X 1)
            gt_novel: ground truth detections for N videos (Dimension: N X 1)

        Returns:
            Dictionary containing indices for novelty introduction and change in world prediction.
        """
        return m_num_stats(p_novel, gt_novel)

    def m_ndp(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp function.

        Args:
            p_novel: detection predictions for N videos (Dimension: N X 1)
            gt_novel: ground truth detections for N videos (Dimension: N X 1)

        Returns:
            Dictionary containing novelty detection performance over the test.
        """
        return m_ndp(p_novel, gt_novel, mode="full_test")

    def m_ndp_pre(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp_pre function.

        Args:
            p_novel: detection predictions for N videos (Dimension: N X 1)
            gt_novel: ground truth detections for N videos (Dimension: N X 1)

        Returns:
            Dictionary containing detection performance pre novelty.
        """
        return m_ndp(p_novel, gt_novel, mode="pre_novelty")

    def m_ndp_post(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp_post function.

        Args:
            p_novel: detection predictions for N videos (Dimension: N X 1)
            gt_novel: ground truth detections for N videos (Dimension: N X 1)

        Returns:
            Dictionary containing detection performance post novelty.
        """
        return m_ndp(p_novel, gt_novel, mode="post_novelty")

    def m_ndp_failed_reaction(
        self,
        p_novel: DataFrame,
        gt_novel: DataFrame,
        p_class: DataFrame,
        gt_class: DataFrame,
    ) -> Dict:
        """
        m_ndp_failed_reaction function.

        Args:
            p_novel: detection predictions for N videos (Dimension: N X 1)
            gt_novel: ground truth detections for N videos (Dimension: N X 1)
            p_class: class predictions with video id for N videos (Dimension: N X 90 [vid,novel_class,88 known class])
            gt_class: ground truth classes for N videos (Dimension: N X 1)

        Returns:
            Dictionary containing TP, FP, TN, FN, top1, top3 accuracy over the test.
        """
        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return m_ndp_failed_reaction(p_novel, gt_novel, class_prob, gt_class_idx)

    def m_accuracy_on_novel(
        self, p_class: DataFrame, gt_class: DataFrame, gt_novel: DataFrame
    ) -> Dict:
        """
        m_accuracy_on_novel function.

        Args:
            p_class: class predictions with video id for N videos (Dimension: N X 90 [vid,novel_class,88 known class])
            gt_class: ground truth classes for N videos (Dimension: N X 1)
            gt_novel: ground truth detections for N videos (Dimension: N X 1)

        Returns:
            Accuracy on novely samples
        """
        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return m_accuracy_on_novel(class_prob, gt_class_idx, gt_novel)

    def m_is_cdt_and_is_early(self, gt_idx: int, ta2_idx: int, test_len: int) -> Dict:
        """
        m_is_cdt_and_is_early function.

        Args:
            gt_idx: Index when novelty is introduced
            ta2_idx: Index when change is detected
            test_len: Length of test

        Returns
            Dictionary containing boolean showing if change was was detected and if it was detected early
        """
        is_cdt = (ta2_idx >= gt_idx) & (ta2_idx < test_len)
        is_early = ta2_idx < gt_idx
        return {"Is CDT": is_cdt, "Is Early": is_early}

    def m_nrp(self, ta2_acc: Dict, baseline_acc: Dict) -> Dict:
        """
        m_nrp function.

        Args:
            ta2_acc: Accuracy scores for the agent
            baseline_acc: Accuracy scores for baseline

        Returns:
            Reaction performance for the agent
        """
        nrp = {}
        nrp["M_nrp_post_top3"] = 100 * (ta2_acc["post_top3"] / baseline_acc["pre_top3"])
        nrp["M_nrp_post_top1"] = 100 * (ta2_acc["post_top1"] / baseline_acc["pre_top1"])
        return nrp
