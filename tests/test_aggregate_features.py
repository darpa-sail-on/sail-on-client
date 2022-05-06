"""Tests for aggregating features."""

from tempfile import TemporaryDirectory
import os
import pickle as pkl


def _get_feature_keys(feature_path):
    """
    Get keys associated with features from the pickle file.

    Args:
        feature_path: Path to a pickle file with features

    Returns:
        Set of keys associated with features
    """
    features = pkl.load(open(feature_path, "rb"))
    return set(features["features_dict"].keys())


def test_aggregate_features(script_runner):
    """
    Test script that aggregate features.

    Args:
        script_runner: Fixture to run scripts provided by pytest-console-scripts

    Returns:
        None
    """
    with TemporaryDirectory() as tempdirname:
        test_path = os.path.dirname(__file__)
        feature_path = os.path.join(test_path, "mock_results", "activity_recognition")
        f1_path = os.path.join(
            feature_path, "OND.2.10006.9373345_timesformer_features.pkl"
        )
        f2_path = os.path.join(
            feature_path, "OND.2.10007.9373345_timesformer_features.pkl"
        )
        op_path = os.path.join(tempdirname, "aggregated_timesformer_features.pkl")
        ret = script_runner.run(
            "aggregate-features",
            "--feature-paths",
            f1_path,
            f2_path,
            "--output-path",
            op_path,
        )
        assert ret.success
        f1_video_ids = _get_feature_keys(f1_path)
        f2_video_ids = _get_feature_keys(f2_path)
        op_video_ids = _get_feature_keys(op_path)
        assert op_video_ids.intersection(f1_video_ids) == f1_video_ids
        assert op_video_ids.intersection(f2_video_ids) == f2_video_ids
