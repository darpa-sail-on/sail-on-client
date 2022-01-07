"""Program Metric Functions."""

import numpy as np
from sail_on_client.evaluate.utils import (
    check_novel_validity,
    check_class_validity,
    get_first_detect_novelty,
    top1_accuracy,
    top3_accuracy,
    get_rolling_stats,
    topk_accuracy,
)


DETECT_THRESH_ = [0.175, 0.225, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _get_threshold(p_novel):
    return np.min(p_novel) + ((np.max(p_novel) - np.min(p_novel)) / 2.0)


def M_num(p_novel, gt_novel):
    """
    Program Metric: Number of samples needed for detecting novelty.
    The method computes number of GT novel samples needed to predict the first true positive.

    Inputs:
    p_novel: NX1 vector with each element corresponding to probability of novelty
    gt_novel: NX1 vector with each element 0 (not novel) or 1 (novel)

    Returns:
    Single scalar for number of GT novel samples
    """
    res = {}
    check_novel_validity(p_novel, gt_novel)
    if np.sum(gt_novel) < 1:
        # only print warning first time.
        # prints too many times for rounds.
        # print('===============TestWarning: No novelty in this test.')
        for thresh in DETECT_THRESH_:
            res[f"{thresh}"] = -1
    else:
        for thresh in DETECT_THRESH_:
            res[f"{thresh}"] = ((p_novel[gt_novel == 1] >= thresh).argmax(axis=0)) + 1
    return res


def M_num_stats(p_novel, gt_novel):
    """
    Program Metric: Number of samples needed for detecting novelty.
    The method computes number of GT novel samples needed to predict the first true positive.

    Inputs:
    p_novel: NX1 vector with each element corresponding to probability of novelty
    gt_novel: NX1 vector with each element 0 (not novel) or 1 (novel)

    Returns:
    Single scalar for number of GT novel samples
    """
    check_novel_validity(p_novel, gt_novel)
    res = {}
    if np.sum(gt_novel) < 1:
        # only print warning first time.
        first_novel_indx = len(gt_novel) + 1
    else:
        first_novel_indx = np.where(gt_novel == 1)[0][0] + 1
    res[f"GT_indx"] = first_novel_indx

    for thresh in DETECT_THRESH_:
        res[f"P_indx_{thresh}"] = get_first_detect_novelty(p_novel, thresh)
    return res


def M_ndp(p_novel, gt_novel, mode="full_test"):
    """
    Program Metric: Novelty detection performance.
    The method computes per-sample novelty detection performance

    Inputs:
    p_novel: NX1 vector with each element corresponding to probability of it being novel
    gt_novel: NX1 vector with each element 0 (not novel) or 1 (novel)
    mode: if 'full_test' computes on all test samples, if 'post_novelty' computes from first GT novel sample

    Returns:
    Dictionary of various metrics: Accuracy, Precision, Recall, F1_score and Confusion matrix
    """
    check_novel_validity(p_novel, gt_novel)
    if mode not in ["full_test", "pre_novelty", "post_novelty"]:
        raise Exception(
            "Mode should be one of ['full_test','pre_novelty', 'post_novelty']"
        )

    def get_all_scores_ndp(thresh):
        add_suffix = lambda x: f"{x}_{thresh}"
        preds = p_novel > thresh
        gt_novel_ = gt_novel
        if mode == "post_novelty":
            if any(gt_novel != 0):
                post_novel_idx = (gt_novel != 0).argmax(axis=0)
                preds = preds[post_novel_idx:]
                gt_novel_ = gt_novel[post_novel_idx:]
            else:
                return {
                    add_suffix("accuracy"): -1,
                    add_suffix("precision"): -1,
                    add_suffix("recall"): -1,
                    add_suffix("f1_score"): -1,
                    add_suffix("TP"): -1,
                    add_suffix("FP"): -1,
                    add_suffix("TN"): -1,
                    add_suffix("FN"): -1,
                }
        elif mode == "pre_novelty":
            if any(gt_novel != 0) and not all(gt_novel != 0):
                post_novel_idx = (gt_novel != 0).argmax(axis=0)
                preds = preds[:post_novel_idx]
                gt_novel_ = gt_novel[:post_novel_idx]

        tp = np.sum(np.logical_and(preds == 1, gt_novel_ == 1))
        fp = np.sum(np.logical_and(preds == 1, gt_novel_ == 0))
        tn = np.sum(np.logical_and(preds == 0, gt_novel_ == 0))
        fn = np.sum(np.logical_and(preds == 0, gt_novel_ == 1))

        acc = (tp + tn) / (tp + tn + fp + fn)
        if tp + fp == 0.0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0.0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision == 0.0 and recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        return {
            add_suffix("accuracy"): round(acc, 5),
            add_suffix("precision"): round(precision, 5),
            add_suffix("recall"): round(recall, 5),
            add_suffix("f1_score"): round(f1_score, 5),
            add_suffix("TP"): tp,
            add_suffix("FP"): fp,
            add_suffix("TN"): tn,
            add_suffix("FN"): fn,
        }

    res = {}
    for thresh in DETECT_THRESH_:
        res.update(get_all_scores_ndp(thresh))
    return res


def M_acc(gt_novel, p_class, gt_class, round_size, asymptotic_start_round):
    """
    Computes top1 and top3 accuracy on:
    1. full_test
    2. pre_novelty (used for M_NRP later)
    3. post_novelty (used for M_NRP later)
    4. asymptotic_100 (last 100 samples; 2% for a 5k test)
    5. asymptotic_250 (last 250 samples; 5% for a 5k test)
    6. asymptotic_500 (last 500 samples; 10% for a 5k test)
    """
    # assert len(gt_novel) > 999, ('Assuming tests are 1000 at least.')
    # full test
    batch_size = round_size
    results = {}
    try:
        results["full_top1"] = top1_accuracy(p_class, gt_class, txt="full_top1")
        results["full_top3"] = top3_accuracy(p_class, gt_class, txt="full_top3")

        # red button push:
        if np.sum(gt_novel) < 1:
            # only print warning first time.
            first_novel_indx = len(gt_novel) + 1
        else:
            first_novel_indx = np.where(gt_novel == 1)[0][0] + 1

        if first_novel_indx == len(gt_novel) + 1:
            # print('===============TestWarning: No novelty in this test.')
            results["pre_top1"] = top1_accuracy(p_class, gt_class, txt="pre_top1")
            results["pre_top3"] = top3_accuracy(p_class, gt_class, txt="pre_top3")
            results["post_top1"] = -1
            results["post_top3"] = -1
            results["post_mean_top1"] = -1
            results["post_mean_top3"] = -1
            results["post_std_top1"] = -1
            results["post_std_top3"] = -1
        else:
            # assert first_novel_indx < len(
            #     gt_novel) - 500, ('Needs to change for asymptotic performance.')
            # pre_novelty
            if first_novel_indx == 0:
                results["pre_top1"] = -1
                results["pre_top3"] = -1
                results["pre_mean_top1"] = -1
                results["pre_mean_top3"] = -1
                results["pre_std_top1"] = -1
                results["pre_std_top3"] = -1
            else:
                p_class_pre = p_class[:first_novel_indx]
                gt_class_pre = gt_class[:first_novel_indx]
                results["pre_top1"] = top1_accuracy(
                    p_class_pre, gt_class_pre, txt="pre_top1"
                )
                results["pre_top3"] = top3_accuracy(
                    p_class_pre, gt_class_pre, txt="pre_top3"
                )
                [results["pre_mean_top1"], results["pre_std_top1"]] = get_rolling_stats(
                    p_class_pre, gt_class_pre, k=1, window_size=batch_size
                )
                [results["pre_mean_top3"], results["pre_std_top3"]] = get_rolling_stats(
                    p_class_pre, gt_class_pre, k=3, window_size=batch_size
                )

            # post_novelty
            p_class_post = p_class[first_novel_indx:]
            gt_class_post = gt_class[first_novel_indx:]
            results["post_top1"] = top1_accuracy(
                p_class_post, gt_class_post, txt="post_top1"
            )
            results["post_top3"] = top3_accuracy(
                p_class_post, gt_class_post, txt="post_top3"
            )
            [results["post_mean_top1"], results["post_std_top1"]] = get_rolling_stats(
                p_class_post, gt_class_post, k=1, window_size=batch_size
            )
            [results["post_mean_top3"], results["post_std_top3"]] = get_rolling_stats(
                p_class_post, gt_class_post, k=3, window_size=batch_size
            )

        # asymptotic performance
        for last_i in np.arange(
            int(asymptotic_start_round) * batch_size, gt_novel.shape[0], round_size
        ):
            if len(gt_novel) > last_i:
                p_class_asym = p_class[-last_i:]
                gt_class_asym = gt_class[-last_i:]
                results[f"asymptotic_{last_i}_top1"] = top1_accuracy(
                    p_class_asym, gt_class_asym, txt=f"asymptotic_{last_i}_top1"
                )
                results[f"asymptotic_{last_i}_top3"] = top3_accuracy(
                    p_class_asym, gt_class_asym, txt=f"asymptotic_{last_i}_top3"
                )
                [
                    results[f"asymptotic_{last_i}_mean_top1"],
                    results[f"asymptotic_{last_i}_std_top1"],
                ] = get_rolling_stats(
                    p_class_asym, gt_class_asym, k=1, window_size=batch_size
                )
                [
                    results[f"asymptotic_{last_i}_mean_top3"],
                    results[f"asymptotic_{last_i}_std_top3"],
                ] = get_rolling_stats(
                    p_class_asym, gt_class_asym, k=3, window_size=batch_size
                )
            else:
                results[f"asymptotic_{last_i}_top1"] = -1
                results[f"asymptotic_{last_i}_top3"] = -1
                results[f"asymptotic_{last_i}_mean_top1"] = -1
                results[f"asymptotic_{last_i}_mean_top3"] = -1
                results[f"asymptotic_{last_i}_std_top1"] = -1
                results[f"asymptotic_{last_i}_std_top3"] = -1
    except Exception as ex:
        import code
        import traceback as tb

        tb.print_stack()
        namespace = globals().copy()
        namespace.update(locals())
        code.interact(local=namespace)

    return results


def M_ndp_pre(p_novel, gt_novel):
    return M_ndp(p_novel, gt_novel, mode="pre_novelty")


def M_ndp_post(p_novel, gt_novel):
    return M_ndp(p_novel, gt_novel, mode="post_novelty")


def M_ndp_failed_reaction(p_novel, gt_novel, p_class, gt_class, mode="full_test"):
    """
    Additional Metric: Novelty detection when reaction fails.
    The method computes novelty detection performance for only on samples with incorrect k-class predictions

    Inputs:
    p_novel: NX1 vector with each element corresponding to probability of novelty
    gt_novel: NX1 vector with each element 0 (not novel) or 1 (novel)
    p_class : Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
    gt_class : Nx1 vector with ground-truth class for each sample
    mode: if 'full_test' computes on all test samples, if 'post_novelty' computes from the first GT novel sample
    k: 'k' used in top-K accuracy

    Returns:
    Dictionary of various metrics: Accuracy, Precision, Recall, F1_score and Confusion matrix
    """
    check_class_validity(p_class, gt_class)
    results = {}
    for k in [1, 3]:
        p_class_k = np.argsort(-p_class, axis=1)[:, :k]
        gt_class_k = gt_class[:, np.newaxis]
        check_zero = p_class_k - gt_class_k
        incorrect_mask = ~np.any(check_zero == 0, axis=1)
        if np.sum(incorrect_mask) == 0:
            warnings.warn(
                "WARNING! No incorrect predictions found. Returning empty dictionary"
            )
            for metric in {
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "FN",
                "TP",
                "FP",
                "TN",
            }:
                results[f"top{k}_{metric}"] = -1
            continue
        p_novel_k = p_novel[incorrect_mask]
        gt_novel_k = gt_novel[incorrect_mask]
        res = M_ndp(p_novel_k, gt_novel_k, mode)
        for metric in res:
            results[f"top{k}_{metric}"] = res[metric]
    return results


def M_accuracy_on_novel(p_class, gt_class, gt_novel):
    """
    Additional Metric: Novelty robustness.
    The method computes top-K accuracy for only the novel samples
    Inputs:
    p_class : Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
    gt_class : Nx1 vector with ground-truth class for each sample
    gt_novel : Nx1 binary vector corresponding to the ground truth novel{1}/seen{0} labels
    k : K value to compute accuracy at
    Returns:
    Accuracy at rank-k
    """
    check_class_validity(p_class, gt_class)
    if np.sum(gt_novel) < 1:
        # print('===============TestWarning: No novelty in this test.')
        return {'top3_acc_novel_only': -1,
                'top1_acc_novel_only': -1}
    p_class = p_class[gt_novel == 1]
    gt_class = gt_class[gt_novel == 1]
    return {'top3_acc_novel_only': topk_accuracy(p_class, gt_class, k=3),
            'top1_acc_novel_only': topk_accuracy(p_class, gt_class, k=1)}
