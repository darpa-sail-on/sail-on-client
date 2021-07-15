"""Json Encoder for Numpy Array."""

import json
import numpy as np
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """An encoder to convert numpy data types to python primitives."""

    def default(self, obj: Any) -> Any:
        """Defaults for numpy types."""
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
