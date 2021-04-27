"""Utility function for sail-on-client."""

import os
import numpy as np
import logging
import json
from sailon_tinker_launcher.deprecated_tinker import harness

from typing import Dict, Any

log = logging.getLogger(__name__)


def safe_remove(file_path: str) -> None:
    """
    Remove a file after checking that it exists.

    Args:
        file_path (str): File path that should be removed

    Return:
        None
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def safe_remove_results(results: dict) -> None:
    """
    Remove results present in a dict.

    Args:
        file_path (str): File path that should be removed

    Return:
        None
    """
    for result_files in results.values():
        safe_remove(result_files)


def update_harness_parameters(ip_harness: harness, new_parameters: Dict) -> harness:
    """
    Update parameters in a harness.

    Args:
        ip_harness (harness): Input harness with old parameters
        new_parameters (dict): Dictionary containing new parameters

    Return:
        Updated harness object
    """
    for param_name, param_value in new_parameters.items():
        if not hasattr(ip_harness, param_name):
            log.warn(
                f"{param_name} is not an attribute in the harness, adding the attribute"
            )
        setattr(ip_harness, param_name, param_value)
    if hasattr(ip_harness, "update_provider"):
        ip_harness.update_provider()
    return ip_harness


class NumpyEncoder(json.JSONEncoder):
    """An encoder to convert numpy data types to python primitives."""

    def default(self, obj: Any) -> Any:
        """
        Serialization defaults for numpy types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)
