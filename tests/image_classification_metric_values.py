"""Expected values of image classfication metrics."""

import pytest


@pytest.fixture(scope="function")
def expected_ic_m_acc_values():
    """Expected Values for m_acc."""
    return {
        "full_top1": 0.39922,
        "full_top3": 0.49062,
        "pre_top1": 0.69161,
        "pre_top3": 0.83577,
        "pre_mean_top1": 0.6991759465478842,
        "pre_std_top1": 0.04058707296133295,
        "pre_mean_top3": 0.84358574610245,
        "pre_std_top3": 0.0272606036370915,
        "post_top1": 0.31958,
        "post_top3": 0.39662,
        "post_mean_top1": 0.32179822268687924,
        "post_std_top1": 0.042994685097466415,
        "post_mean_top3": 0.39578672242550966,
        "post_std_top3": 0.041940639603993336,
        "asymptotic_500_top1": 0.296,
        "asymptotic_500_top3": 0.392,
        "asymptotic_500_mean_top1": 0.29077306733167085,
        "asymptotic_500_std_top1": 0.023057555655748834,
        "asymptotic_500_mean_top3": 0.3889027431421447,
        "asymptotic_500_std_top3": 0.023308646838809823,
        "asymptotic_600_top1": 0.305,
        "asymptotic_600_top3": 0.39333,
        "asymptotic_600_mean_top1": 0.2927145708582835,
        "asymptotic_600_std_top1": 0.024162292260161675,
        "asymptotic_600_mean_top3": 0.3813373253493014,
        "asymptotic_600_std_top3": 0.027618978692774576,
        "asymptotic_700_top1": 0.30857,
        "asymptotic_700_top3": 0.39143,
        "asymptotic_700_mean_top1": 0.302495840266223,
        "asymptotic_700_std_top1": 0.032037900461814116,
        "asymptotic_700_mean_top3": 0.3871214642262895,
        "asymptotic_700_std_top3": 0.02985297676702389,
        "asymptotic_800_top1": 0.3125,
        "asymptotic_800_top3": 0.39125,
        "asymptotic_800_mean_top1": 0.30406562054208275,
        "asymptotic_800_std_top1": 0.030624756126260803,
        "asymptotic_800_mean_top3": 0.38296718972895866,
        "asymptotic_800_std_top3": 0.030421520704071274,
        "asymptotic_900_top1": 0.31556,
        "asymptotic_900_top3": 0.39667,
        "asymptotic_900_mean_top1": 0.31213483146067417,
        "asymptotic_900_std_top1": 0.03677379226776603,
        "asymptotic_900_mean_top3": 0.3895630461922597,
        "asymptotic_900_std_top3": 0.03437562847002384,
        "asymptotic_1000_top1": 0.318,
        "asymptotic_1000_top3": 0.402,
        "asymptotic_1000_mean_top1": 0.3143285238623752,
        "asymptotic_1000_std_top1": 0.035592114559528445,
        "asymptotic_1000_mean_top3": 0.3971476137624862,
        "asymptotic_1000_std_top3": 0.03915452815263005,
        "asymptotic_1100_top1": 0.31636,
        "asymptotic_1100_top3": 0.39636,
        "asymptotic_1100_mean_top1": 0.3139260739260739,
        "asymptotic_1100_std_top1": 0.03432303205532007,
        "asymptotic_1100_mean_top3": 0.3945654345654345,
        "asymptotic_1100_std_top3": 0.039418726058004006,
        "asymptotic_1200_top1": 0.31833,
        "asymptotic_1200_top3": 0.39583,
        "asymptotic_1200_mean_top1": 0.3153678474114442,
        "asymptotic_1200_std_top1": 0.03377433159016104,
        "asymptotic_1200_mean_top3": 0.3931062670299728,
        "asymptotic_1200_std_top3": 0.0386237864388054,
        "asymptotic_1300_top1": 0.31385,
        "asymptotic_1300_top3": 0.39,
        "asymptotic_1300_mean_top1": 0.31247293921731895,
        "asymptotic_1300_std_top1": 0.03442328875587672,
        "asymptotic_1300_mean_top3": 0.38788509575353874,
        "asymptotic_1300_std_top3": 0.04147316545283617,
        "asymptotic_1400_top1": 0.31071,
        "asymptotic_1400_top3": 0.38786,
        "asymptotic_1400_mean_top1": 0.30975403535741736,
        "asymptotic_1400_std_top1": 0.034626815879230555,
        "asymptotic_1400_mean_top3": 0.38428132205995386,
        "asymptotic_1400_std_top3": 0.04199499087460914,
        "asymptotic_1500_top1": 0.312,
        "asymptotic_1500_top3": 0.388,
        "asymptotic_1500_mean_top1": 0.3106209850107067,
        "asymptotic_1500_std_top1": 0.03383531273516182,
        "asymptotic_1500_mean_top3": 0.38536045681655967,
        "asymptotic_1500_std_top3": 0.040831036596019145,
        "asymptotic_1600_top1": 0.3175,
        "asymptotic_1600_top3": 0.39313,
        "asymptotic_1600_mean_top1": 0.31253164556962026,
        "asymptotic_1600_std_top1": 0.03493784716545886,
        "asymptotic_1600_mean_top3": 0.3879013990672885,
        "asymptotic_1600_std_top3": 0.04191809002536541,
        "asymptotic_1700_top1": 0.32059,
        "asymptotic_1700_top3": 0.39353,
        "asymptotic_1700_mean_top1": 0.31893816364772015,
        "asymptotic_1700_std_top1": 0.0423172163412919,
        "asymptotic_1700_mean_top3": 0.39204247345409116,
        "asymptotic_1700_std_top3": 0.04414479805003304,
        "asymptotic_1800_top1": 0.32444,
        "asymptotic_1800_top3": 0.39667,
        "asymptotic_1800_mean_top1": 0.3229100529100529,
        "asymptotic_1800_std_top1": 0.04428978927669073,
        "asymptotic_1800_mean_top3": 0.3939153439153439,
        "asymptotic_1800_std_top3": 0.04367206543913269,
        "asymptotic_1900_top1": 0.32368,
        "asymptotic_1900_top3": 0.39737,
        "asymptotic_1900_mean_top1": 0.32283176013325926,
        "asymptotic_1900_std_top1": 0.04349558102280415,
        "asymptotic_1900_mean_top3": 0.3945696835091616,
        "asymptotic_1900_std_top3": 0.04270890815690953,
        "asymptotic_2000_top1": 0.321,
        "asymptotic_2000_top3": 0.3975,
        "asymptotic_2000_mean_top1": 0.3221988427143609,
        "asymptotic_2000_std_top1": 0.042830196169561754,
        "asymptotic_2000_mean_top3": 0.3957548658600737,
        "asymptotic_2000_std_top3": 0.042069127119232004,
        "asymptotic_2100_top1": 0.33524,
        "asymptotic_2100_top3": 0.41476,
        "asymptotic_2100_mean_top1": 0.3254772613693153,
        "asymptotic_2100_std_top1": 0.050888943865010125,
        "asymptotic_2100_mean_top3": 0.4025537231384308,
        "asymptotic_2100_std_top3": 0.05745628979825715,
        "asymptotic_2200_top1": 0.34773,
        "asymptotic_2200_top3": 0.43227,
        "asymptotic_2200_mean_top1": 0.34077106139933366,
        "asymptotic_2200_std_top1": 0.08473642855385971,
        "asymptotic_2200_mean_top3": 0.4228177058543551,
        "asymptotic_2200_std_top3": 0.10676614226416765,
        "asymptotic_2300_top1": 0.36435,
        "asymptotic_2300_top3": 0.45,
        "asymptotic_2300_mean_top1": 0.35622898682417087,
        "asymptotic_2300_std_top1": 0.10920643627102965,
        "asymptotic_2300_mean_top3": 0.4407042253521127,
        "asymptotic_2300_std_top3": 0.13271844506100505,
        "asymptotic_2400_top1": 0.37958,
        "asymptotic_2400_top3": 0.4675,
        "asymptotic_2400_mean_top1": 0.372103433289874,
        "asymptotic_2400_std_top1": 0.13025096074564638,
        "asymptotic_2400_mean_top3": 0.45810517166449366,
        "asymptotic_2400_std_top3": 0.15339539496015317,
        "asymptotic_2500_top1": 0.3928,
        "asymptotic_2500_top3": 0.4832,
        "asymptotic_2500_mean_top1": 0.3872386505622657,
        "asymptotic_2500_std_top1": 0.1468017260531301,
        "asymptotic_2500_mean_top3": 0.47534360683048726,
        "asymptotic_2500_std_top3": 0.17148146343502754,
    }


@pytest.fixture(scope="function")
def expected_ic_m_acc_roundwise_values():
    """Expected m_acc value for a round."""
    return {
        "top1_accuracy_round_0": 0.39922,
        "top3_accuracy_round_0": 0.49062,
    }


@pytest.fixture(scope="function")
def expected_ic_m_num_values():
    """Expected Values for m_num."""
    return {
        "0.175": 1,
        "0.225": 1,
        "0.3": 1,
        "0.4": 1,
        "0.5": 1,
        "0.6": 1,
        "0.7": 1,
        "0.8": 1,
        "0.9": 1,
    }


@pytest.fixture(scope="function")
def expected_ic_m_num_stats_values():
    """Expected Values for m_num_stats."""
    return {
        "GT_indx": 548,
        "P_indx_0.175": 548,
        "P_indx_0.225": 548,
        "P_indx_0.3": 548,
        "P_indx_0.4": 548,
        "P_indx_0.5": 548,
        "P_indx_0.6": 548,
        "P_indx_0.7": 548,
        "P_indx_0.8": 548,
        "P_indx_0.9": 548,
    }


@pytest.fixture(scope="function")
def expected_ic_m_ndp_values():
    """Expected Values for m_ndp."""
    return {
        "accuracy_0.175": 1.0,
        "precision_0.175": 1.0,
        "recall_0.175": 1.0,
        "f1_score_0.175": 1.0,
        "TP_0.175": 2013,
        "FP_0.175": 0,
        "TN_0.175": 547,
        "FN_0.175": 0,
        "accuracy_0.225": 1.0,
        "precision_0.225": 1.0,
        "recall_0.225": 1.0,
        "f1_score_0.225": 1.0,
        "TP_0.225": 2013,
        "FP_0.225": 0,
        "TN_0.225": 547,
        "FN_0.225": 0,
        "accuracy_0.3": 1.0,
        "precision_0.3": 1.0,
        "recall_0.3": 1.0,
        "f1_score_0.3": 1.0,
        "TP_0.3": 2013,
        "FP_0.3": 0,
        "TN_0.3": 547,
        "FN_0.3": 0,
        "accuracy_0.4": 1.0,
        "precision_0.4": 1.0,
        "recall_0.4": 1.0,
        "f1_score_0.4": 1.0,
        "TP_0.4": 2013,
        "FP_0.4": 0,
        "TN_0.4": 547,
        "FN_0.4": 0,
        "accuracy_0.5": 1.0,
        "precision_0.5": 1.0,
        "recall_0.5": 1.0,
        "f1_score_0.5": 1.0,
        "TP_0.5": 2013,
        "FP_0.5": 0,
        "TN_0.5": 547,
        "FN_0.5": 0,
        "accuracy_0.6": 1.0,
        "precision_0.6": 1.0,
        "recall_0.6": 1.0,
        "f1_score_0.6": 1.0,
        "TP_0.6": 2013,
        "FP_0.6": 0,
        "TN_0.6": 547,
        "FN_0.6": 0,
        "accuracy_0.7": 1.0,
        "precision_0.7": 1.0,
        "recall_0.7": 1.0,
        "f1_score_0.7": 1.0,
        "TP_0.7": 2013,
        "FP_0.7": 0,
        "TN_0.7": 547,
        "FN_0.7": 0,
        "accuracy_0.8": 1.0,
        "precision_0.8": 1.0,
        "recall_0.8": 1.0,
        "f1_score_0.8": 1.0,
        "TP_0.8": 2013,
        "FP_0.8": 0,
        "TN_0.8": 547,
        "FN_0.8": 0,
        "accuracy_0.9": 1.0,
        "precision_0.9": 1.0,
        "recall_0.9": 1.0,
        "f1_score_0.9": 1.0,
        "TP_0.9": 2013,
        "FP_0.9": 0,
        "TN_0.9": 547,
        "FN_0.9": 0,
    }


@pytest.fixture(scope="function")
def expected_ic_m_ndp_pre_values():
    """Expected Values for m_ndp_pre."""
    return {
        "accuracy_0.175": 1.0,
        "precision_0.175": 0.0,
        "recall_0.175": 0.0,
        "f1_score_0.175": 0.0,
        "TP_0.175": 0,
        "FP_0.175": 0,
        "TN_0.175": 547,
        "FN_0.175": 0,
        "accuracy_0.225": 1.0,
        "precision_0.225": 0.0,
        "recall_0.225": 0.0,
        "f1_score_0.225": 0.0,
        "TP_0.225": 0,
        "FP_0.225": 0,
        "TN_0.225": 547,
        "FN_0.225": 0,
        "accuracy_0.3": 1.0,
        "precision_0.3": 0.0,
        "recall_0.3": 0.0,
        "f1_score_0.3": 0.0,
        "TP_0.3": 0,
        "FP_0.3": 0,
        "TN_0.3": 547,
        "FN_0.3": 0,
        "accuracy_0.4": 1.0,
        "precision_0.4": 0.0,
        "recall_0.4": 0.0,
        "f1_score_0.4": 0.0,
        "TP_0.4": 0,
        "FP_0.4": 0,
        "TN_0.4": 547,
        "FN_0.4": 0,
        "accuracy_0.5": 1.0,
        "precision_0.5": 0.0,
        "recall_0.5": 0.0,
        "f1_score_0.5": 0.0,
        "TP_0.5": 0,
        "FP_0.5": 0,
        "TN_0.5": 547,
        "FN_0.5": 0,
        "accuracy_0.6": 1.0,
        "precision_0.6": 0.0,
        "recall_0.6": 0.0,
        "f1_score_0.6": 0.0,
        "TP_0.6": 0,
        "FP_0.6": 0,
        "TN_0.6": 547,
        "FN_0.6": 0,
        "accuracy_0.7": 1.0,
        "precision_0.7": 0.0,
        "recall_0.7": 0.0,
        "f1_score_0.7": 0.0,
        "TP_0.7": 0,
        "FP_0.7": 0,
        "TN_0.7": 547,
        "FN_0.7": 0,
        "accuracy_0.8": 1.0,
        "precision_0.8": 0.0,
        "recall_0.8": 0.0,
        "f1_score_0.8": 0.0,
        "TP_0.8": 0,
        "FP_0.8": 0,
        "TN_0.8": 547,
        "FN_0.8": 0,
        "accuracy_0.9": 1.0,
        "precision_0.9": 0.0,
        "recall_0.9": 0.0,
        "f1_score_0.9": 0.0,
        "TP_0.9": 0,
        "FP_0.9": 0,
        "TN_0.9": 547,
        "FN_0.9": 0,
    }


@pytest.fixture(scope="function")
def expected_ic_m_ndp_post_values():
    """Expected Values for m_ndp_post."""
    return {
        "accuracy_0.175": 1.0,
        "precision_0.175": 1.0,
        "recall_0.175": 1.0,
        "f1_score_0.175": 1.0,
        "TP_0.175": 2013,
        "FP_0.175": 0,
        "TN_0.175": 0,
        "FN_0.175": 0,
        "accuracy_0.225": 1.0,
        "precision_0.225": 1.0,
        "recall_0.225": 1.0,
        "f1_score_0.225": 1.0,
        "TP_0.225": 2013,
        "FP_0.225": 0,
        "TN_0.225": 0,
        "FN_0.225": 0,
        "accuracy_0.3": 1.0,
        "precision_0.3": 1.0,
        "recall_0.3": 1.0,
        "f1_score_0.3": 1.0,
        "TP_0.3": 2013,
        "FP_0.3": 0,
        "TN_0.3": 0,
        "FN_0.3": 0,
        "accuracy_0.4": 1.0,
        "precision_0.4": 1.0,
        "recall_0.4": 1.0,
        "f1_score_0.4": 1.0,
        "TP_0.4": 2013,
        "FP_0.4": 0,
        "TN_0.4": 0,
        "FN_0.4": 0,
        "accuracy_0.5": 1.0,
        "precision_0.5": 1.0,
        "recall_0.5": 1.0,
        "f1_score_0.5": 1.0,
        "TP_0.5": 2013,
        "FP_0.5": 0,
        "TN_0.5": 0,
        "FN_0.5": 0,
        "accuracy_0.6": 1.0,
        "precision_0.6": 1.0,
        "recall_0.6": 1.0,
        "f1_score_0.6": 1.0,
        "TP_0.6": 2013,
        "FP_0.6": 0,
        "TN_0.6": 0,
        "FN_0.6": 0,
        "accuracy_0.7": 1.0,
        "precision_0.7": 1.0,
        "recall_0.7": 1.0,
        "f1_score_0.7": 1.0,
        "TP_0.7": 2013,
        "FP_0.7": 0,
        "TN_0.7": 0,
        "FN_0.7": 0,
        "accuracy_0.8": 1.0,
        "precision_0.8": 1.0,
        "recall_0.8": 1.0,
        "f1_score_0.8": 1.0,
        "TP_0.8": 2013,
        "FP_0.8": 0,
        "TN_0.8": 0,
        "FN_0.8": 0,
        "accuracy_0.9": 1.0,
        "precision_0.9": 1.0,
        "recall_0.9": 1.0,
        "f1_score_0.9": 1.0,
        "TP_0.9": 2013,
        "FP_0.9": 0,
        "TN_0.9": 0,
        "FN_0.9": 0,
    }


@pytest.fixture(scope="function")
def expected_ic_m_ndp_failed_values():
    """Expected Values for m_ndp_failed."""
    return {
        "top1_accuracy_0.175": 1.0,
        "top1_precision_0.175": 1.0,
        "top1_recall_0.175": 1.0,
        "top1_f1_score_0.175": 1.0,
        "top1_TP_0.175": 1370,
        "top1_FP_0.175": 0,
        "top1_TN_0.175": 168,
        "top1_FN_0.175": 0,
        "top1_accuracy_0.225": 1.0,
        "top1_precision_0.225": 1.0,
        "top1_recall_0.225": 1.0,
        "top1_f1_score_0.225": 1.0,
        "top1_TP_0.225": 1370,
        "top1_FP_0.225": 0,
        "top1_TN_0.225": 168,
        "top1_FN_0.225": 0,
        "top1_accuracy_0.3": 1.0,
        "top1_precision_0.3": 1.0,
        "top1_recall_0.3": 1.0,
        "top1_f1_score_0.3": 1.0,
        "top1_TP_0.3": 1370,
        "top1_FP_0.3": 0,
        "top1_TN_0.3": 168,
        "top1_FN_0.3": 0,
        "top1_accuracy_0.4": 1.0,
        "top1_precision_0.4": 1.0,
        "top1_recall_0.4": 1.0,
        "top1_f1_score_0.4": 1.0,
        "top1_TP_0.4": 1370,
        "top1_FP_0.4": 0,
        "top1_TN_0.4": 168,
        "top1_FN_0.4": 0,
        "top1_accuracy_0.5": 1.0,
        "top1_precision_0.5": 1.0,
        "top1_recall_0.5": 1.0,
        "top1_f1_score_0.5": 1.0,
        "top1_TP_0.5": 1370,
        "top1_FP_0.5": 0,
        "top1_TN_0.5": 168,
        "top1_FN_0.5": 0,
        "top1_accuracy_0.6": 1.0,
        "top1_precision_0.6": 1.0,
        "top1_recall_0.6": 1.0,
        "top1_f1_score_0.6": 1.0,
        "top1_TP_0.6": 1370,
        "top1_FP_0.6": 0,
        "top1_TN_0.6": 168,
        "top1_FN_0.6": 0,
        "top1_accuracy_0.7": 1.0,
        "top1_precision_0.7": 1.0,
        "top1_recall_0.7": 1.0,
        "top1_f1_score_0.7": 1.0,
        "top1_TP_0.7": 1370,
        "top1_FP_0.7": 0,
        "top1_TN_0.7": 168,
        "top1_FN_0.7": 0,
        "top1_accuracy_0.8": 1.0,
        "top1_precision_0.8": 1.0,
        "top1_recall_0.8": 1.0,
        "top1_f1_score_0.8": 1.0,
        "top1_TP_0.8": 1370,
        "top1_FP_0.8": 0,
        "top1_TN_0.8": 168,
        "top1_FN_0.8": 0,
        "top1_accuracy_0.9": 1.0,
        "top1_precision_0.9": 1.0,
        "top1_recall_0.9": 1.0,
        "top1_f1_score_0.9": 1.0,
        "top1_TP_0.9": 1370,
        "top1_FP_0.9": 0,
        "top1_TN_0.9": 168,
        "top1_FN_0.9": 0,
        "top3_accuracy_0.175": 1.0,
        "top3_precision_0.175": 1.0,
        "top3_recall_0.175": 1.0,
        "top3_f1_score_0.175": 1.0,
        "top3_TP_0.175": 1215,
        "top3_FP_0.175": 0,
        "top3_TN_0.175": 89,
        "top3_FN_0.175": 0,
        "top3_accuracy_0.225": 1.0,
        "top3_precision_0.225": 1.0,
        "top3_recall_0.225": 1.0,
        "top3_f1_score_0.225": 1.0,
        "top3_TP_0.225": 1215,
        "top3_FP_0.225": 0,
        "top3_TN_0.225": 89,
        "top3_FN_0.225": 0,
        "top3_accuracy_0.3": 1.0,
        "top3_precision_0.3": 1.0,
        "top3_recall_0.3": 1.0,
        "top3_f1_score_0.3": 1.0,
        "top3_TP_0.3": 1215,
        "top3_FP_0.3": 0,
        "top3_TN_0.3": 89,
        "top3_FN_0.3": 0,
        "top3_accuracy_0.4": 1.0,
        "top3_precision_0.4": 1.0,
        "top3_recall_0.4": 1.0,
        "top3_f1_score_0.4": 1.0,
        "top3_TP_0.4": 1215,
        "top3_FP_0.4": 0,
        "top3_TN_0.4": 89,
        "top3_FN_0.4": 0,
        "top3_accuracy_0.5": 1.0,
        "top3_precision_0.5": 1.0,
        "top3_recall_0.5": 1.0,
        "top3_f1_score_0.5": 1.0,
        "top3_TP_0.5": 1215,
        "top3_FP_0.5": 0,
        "top3_TN_0.5": 89,
        "top3_FN_0.5": 0,
        "top3_accuracy_0.6": 1.0,
        "top3_precision_0.6": 1.0,
        "top3_recall_0.6": 1.0,
        "top3_f1_score_0.6": 1.0,
        "top3_TP_0.6": 1215,
        "top3_FP_0.6": 0,
        "top3_TN_0.6": 89,
        "top3_FN_0.6": 0,
        "top3_accuracy_0.7": 1.0,
        "top3_precision_0.7": 1.0,
        "top3_recall_0.7": 1.0,
        "top3_f1_score_0.7": 1.0,
        "top3_TP_0.7": 1215,
        "top3_FP_0.7": 0,
        "top3_TN_0.7": 89,
        "top3_FN_0.7": 0,
        "top3_accuracy_0.8": 1.0,
        "top3_precision_0.8": 1.0,
        "top3_recall_0.8": 1.0,
        "top3_f1_score_0.8": 1.0,
        "top3_TP_0.8": 1215,
        "top3_FP_0.8": 0,
        "top3_TN_0.8": 89,
        "top3_FN_0.8": 0,
        "top3_accuracy_0.9": 1.0,
        "top3_precision_0.9": 1.0,
        "top3_recall_0.9": 1.0,
        "top3_f1_score_0.9": 1.0,
        "top3_TP_0.9": 1215,
        "top3_FP_0.9": 0,
        "top3_TN_0.9": 89,
        "top3_FN_0.9": 0,
    }


@pytest.fixture(scope="function")
def expected_ic_m_nrp_values():
    """Expected Values for m_nrp."""
    return {"M_nrp_post_top3": 43.29629063598454, "M_nrp_post_top1": 39.80222189010113}
