"""Image Classification Class for metrics for sail-on."""

from sail_on_client.evaluate.program_metrics import ProgramMetrics
from sail_on_client.evaluate.metrics import m_acc, m_num, m_ndp, m_num_stats
from sail_on_client.evaluate.metrics import m_ndp_failed_reaction
from sail_on_client.evaluate.metrics import m_accuracy_on_novel
from sail_on_client.evaluate.utils import topk_accuracy

import numpy as np
from pandas import DataFrame
import pandas as pd

from typing import Dict


class ImageClassificationMetrics(ProgramMetrics):
    """Image Classification program metric class."""

    def __init__(
        self, protocol: str, image_id: int, detection: int, classification: int
    ) -> None:
        """
        Initialize.

        Args:
            protocol: Name of the protocol.
            image_id: Column id for image
            detection: Column id for predicting sample wise world detection
            classification: Column id for predicting sample wise classes

        Returns:
            None
        """
        super().__init__(protocol)
        self.image_id = image_id
        self.detection_id = detection
        self.classification_id = classification
        self.novel_id = -1  # Added only for fixing typechecking problems

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
            gt_novel: ground truth detections (Dimension: [img X detection])
            p_class: class predictions (Dimension: [img X prob that sample is novel, prob of 88 known classes])
            gt_class: ground truth classes (Dimension: [img X detection, classification])
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
            p_class: class predictions
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

        A Program Metric where the number of samples needed for detecting novelty.
        The method computes the number of GT novel samples needed to predict the first true positive.

        Args:
            p_novel: detection predictions (Dimension: [img X novel])
                Nx1 vector with each element corresponding to probability of novelty
            gt_novel: ground truth detections (Dimension: [img X detection])
                Nx1 vector with each element 0 (not novel) or 1 (novel)

        Returns:
            Difference between the novelty introduction and predicting change in world.
        """
        return m_num(p_novel, gt_novel)

    def m_num_stats(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        Program Metric.

        Number of samples needed for detecting novelty.  The method computes number of GT novel
            samples needed to predict the first true positive.

        Args:
            p_novel: detection predictions (Dimension: [img X novel])
                Nx1 vector with each element corresponding to probability of novelty
            gt_novel: ground truth detections (Dimension: [img X detection])
                Nx1 vector with each element 0 (not novel) or 1 (novel)

        Returns:
            Dictionary containing indices for novelty introduction and change in world prediction.
        """
        return m_num_stats(p_novel, gt_novel)

    def m_ndp(
        self, p_novel: np.ndarray, gt_novel: np.ndarray, mode: str = "full_test"
    ) -> Dict:
        """
        Novelty Detection Performance: Program Metric.

            Novelty detection performance.  The method computes per-sample novelty detection performance.

        Args:
            p_novel: detection predictions (Dimension: [img X novel])
                Nx1 vector with each element corresponding to probability of it being novel
            gt_novel: ground truth detections (Dimension: [img X detection])
                Nx1 vector with each element 0 (not novel) or 1 (novel)
            mode: the mode to compute the test.  if 'full_test' computes on all test samples,
                if 'post_novelty' computes from first GT novel sample.  If 'pre_novelty', only calculate
                before first novel sample.

        Returns:
            Dictionary containing novelty detection performance over the test.
        """
        return m_ndp(p_novel, gt_novel, mode="full_test")

    def m_ndp_pre(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        Novelty Detection Performance Pre Red Light.  m_ndp_pre function.

        See :func:`~sail-on-client.evaluation.ImageClassificationMetrics.m_ndp` with
            post_novelty.  This computes to the first GT novel sample.  It really isn't useful
            and is just added for completion.  Should always be 0 since no possible TP.

        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])

        Returns:
            Dictionary containing detection performance pre novelty.
        """
        return m_ndp(p_novel, gt_novel, mode="pre_novelty")

    def m_ndp_post(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        Novelty Detection Performance Post Red Light.  m_ndp_post function.

        See :func:`~sail-on-client.evaluation.ImageClassificationMetrics.m_ndp` with
            post_novelty.  This computes from the first GT novel sample

        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])

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
        mode: str = "full_test",
    ) -> Dict:
        """
        Additional Metric: Novelty detection when reaction fails.

        Not Implemented since no gt_class info for novel samples

        The method computes novelty detection performance for only on samples with incorrect k-class predictions

        Args:
            p_novel: detection predictions (Dimension: [img X novel])
                Nx1 vector with each element corresponding to probability of novelty
            gt_novel: ground truth detections (Dimension: [img X detection])
                Nx1 vector with each element 0 (not novel) or 1 (novel)
            p_class: detection predictions (Dimension: [img X prob that sample is novel, prob of 88 known classes])
                Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
            gt_class: ground truth classes (Dimension:  [img X classification])
                Nx1 vector with ground-truth class for each sample
            mode: if 'full_test' computes on all test samples, if 'post_novelty' computes from the first GT
                novel sample.  If 'pre_novelty', than everything before novelty introduced.

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
        Additional Metric: Novelty robustness.

        The method computes top-K accuracy for only the novel samples

        Args:
            p_class: detection predictions (Dimension: [img X  prob that sample is novel, prob of known classes])
                Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
            gt_class: ground truth classes (Dimension: [img X classification])
                Nx1 vector with ground-truth class for each sample
            gt_novel: ground truth detections (Dimension: N X [img, classification])
                Nx1 binary vector corresponding to the ground truth novel{1}/seen{0} labels

        Returns:
            Accuracy on novely samples
        """

        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()

        return m_accuracy_on_novel(class_prob, gt_class_idx, gt_novel)

    def m_is_cdt_and_is_early(self, gt_idx: int, ta2_idx: int, test_len: int) -> Dict:
        """
        Is change detection and is change detection early (m_is_cdt_and_is_early) function.

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


def convert_df(old_filepath: str, new_filepath: str) -> None:
    """
    Convert from the old df to the new df.

    Args:
        old_filepath:  the filepath the the old *_single_df.csv file
        new_filepath:  the filepath the the old *_single_df.csv file

    Return:
        None
    """
    df = pd.read_csv(old_filepath)
    df["id"] = df.current_path
    df["detection"] = df.cls_novelty.cummax()

    df["classification"] = df.class_id

    # Retain the class num for novel classes but as negative numbers
    mask = df.classification == -1
    df.loc[mask, "classification"] = df.loc[mask, "current_path"].map(
        lambda x: -int(x.split("/")[-2])
    )

    df[["id", "detection", "classification"]].to_csv(new_filepath, index=False)
