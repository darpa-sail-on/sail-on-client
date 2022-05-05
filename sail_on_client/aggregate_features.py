"""Helper script to aggregate features generated by client."""
import argparse
from argparse import Namespace
import pickle as pkl
from tqdm import tqdm
from typing import Dict


def cmd_opts() -> Namespace:
    """
    Command line options for the helper script.

    Args:
        None

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        "Aggregate features generated by the client. Note: The script only supports aggregation for the top level keys."
    )
    parser.add_argument(
        "--feature-paths",
        nargs="+",
        help="Path to feature files that would be aggregated",
    )
    parser.add_argument("--output-path", required=True, help="Path of the output file")
    args = parser.parse_args()
    return args


def main() -> None:
    """
    Entrypoint for the helper script.

    Args:
        None

    Returns:
        None
    """
    args = cmd_opts()
    feature_paths, output_path = args.feature_paths, args.output_path
    features = list(
        map(lambda feature_path: pkl.load(open(feature_path, "rb")), feature_paths)
    )
    aggregated_features: Dict = {}
    for feature in tqdm(features):
        feature_keys = feature.keys()
        for feature_key in feature_keys:
            if feature_key in aggregated_features.keys():
                aggregated_features[feature_key].update(feature[feature_key])
            else:
                aggregated_features[feature_key] = feature[feature_key]
    pkl.dump(aggregated_features, open(output_path, "wb"))


if __name__ == "__main__":
    main()
