"""Helper functions for metrics."""

import numpy as np
import pandas as pd


def check_novel_validity(p_novel, gt_novel):
    """
    Check the validity of the inputs for per-sample novelty detection.

    Inputs:
    p_novel: NX1 vector with each element corresponding to probability of novelty
    gt_novel: NX1 vector with each element 0 (not novel) or 1 (novel)
    """
    if p_novel.shape[0] != gt_novel.shape[0]:
        raise Exception(
            "Number of predicted samples not equal to number of groundtruth samples!"
        )
    if p_novel.ndim != 1:
        raise Exception(
            "Predicted probabilities must be a vector but is an array of dimension {}!".format(
                p_novel.ndim
            )
        )
    if np.any(p_novel < 0) or np.any(p_novel > 1):
        raise Exception("Predicted novel probabilities should be between 0 and 1!")
    if np.any(np.logical_and(gt_novel != 0, gt_novel != 1)):
        raise Exception(
            "Groundtruth array should only consist of 0s (non-novel) or 1s(novel)!"
        )
    return


def check_class_validity(p_class, gt_class):
    """
    Check the validity of the inputs for image classification.

    Inputs:
    p_class: Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
    gt_class: Nx1 vector with ground-truth class for each sample
    """
    if p_class.shape[0] != gt_class.shape[0]:
        raise Exception(
            "Number of predicted samples not equal to number of groundtruth samples!"
        )
    if p_class.shape[0] != gt_class.shape[0]:
        raise Exception(
            "Shape mismatch between class probabilities and novel probabilities!"
        )
    if np.any(p_class < 0) or np.any(p_class > 1):
        raise Exception("Predicted class probabilities should be between 0 and 1!")
    if p_class.ndim != 2:
        raise Exception(
            "Predicted probabilities must be a 2D matrix but is an array of dimension {}!".format(
                p_class.ndim
            )
        )
    if np.max(gt_class) >= p_class.shape[1] or np.min(gt_class < 0):
        raise Exception(
            "Groundtruth class labels must lie in the range [0-{}]!".format(
                p_class.shape[1]
            )
        )
    return


def topk_accuracy(p_class, gt_class, k, txt=""):
    """
    The method computes top-K accuracy.
    Inputs:
    p_class: Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
    gt_class: Nx1 computevector with ground-truth class for each sample
    k: 'k' used in top-K accuracy
    Returns:
    top-K accuracy
    """
    check_class_validity(p_class, gt_class)
    p_class = np.argsort(-p_class)[:, :k]
    gt_class = gt_class[:, np.newaxis]
    check_zero = p_class - gt_class
    correct = np.sum(np.any(check_zero == 0, axis=1).astype(int))
    return round(float(correct) / p_class.shape[0], 5)


def top3_accuracy(p_class, gt_class, txt=""):
    """The method computes top-3 accuracy. (see topk_accuracy() for details)"""
    return topk_accuracy(p_class, gt_class, k=3, txt=txt)


def top1_accuracy(p_class, gt_class, txt=""):
    """The method computes top-1 accuracy. (see topk_accuracy() for details)"""
    return topk_accuracy(p_class, gt_class, k=1, txt=txt)


# compute information for the robustness measures
def get_rolling_stats(p_class, gt_class, k=1, window_size=50):
    """The method computes rolling statistics which are used for robustness measures"""
    p_cls_topk = np.argsort(-p_class)[:, :k]
    gt_cls_indx = gt_class[:, np.newaxis]
    check_zero_topk = p_cls_topk - gt_cls_indx
    topk_correct = 1 * (np.any(check_zero_topk == 0, axis=1))
    acc_mean = pd.Series(topk_correct).rolling(window=window_size).mean().mean()
    acc_std = pd.Series(topk_correct).rolling(window=window_size).mean().std()
    return [acc_mean, acc_std]


def best_fit_slope(xs, ys):
    # https://pythonprogramming.net/how-to-program-best-fit-line-slope-machine-learning-tutorial/
    m = ((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / (
        (np.mean(xs) ** 2) - np.mean(xs ** 2)
    )
    return m


def running_average_cumsum(seq, window=100):
    s = np.insert(np.cumsum(seq), 0, [0])
    return (s[window:] - s[:-window]) * (1.0 / window)


def get_first_detect_novelty(p_novel, thresh):
    if np.sum(p_novel >= thresh) < 1:
        first_detect_novelty = len(p_novel) + 1
    else:
        first_detect_novelty = np.where(p_novel >= thresh)[0][0] + 1
    return first_detect_novelty
