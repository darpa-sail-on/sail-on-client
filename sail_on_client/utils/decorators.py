"""Decorators for sail-on-client."""

import functools
from typing import Any, AnyStr, Callable


def skip_stage(stage_name: str, skip_return: Any = None) -> Callable:
    """
    Skip stages in the protocol.

    Args:
        stage_name: Name of the stage that is covered by the decorated function
        skip_stages: List of stages that should be skipped
        skip_return: Optional return types when the stage is skipped

    Returns:
        Decorated function call
    """

    def skip_stage_decorator(stage_fn: Callable) -> Callable:
        """
        Capture the stage function.

        Args:
            stage_fn: The callable function that would be wrapped

        Returns:
            Wrapped function
        """

        @functools.wraps(stage_fn)
        def skip_stage_fn(self: Any, *args: AnyStr, **kwargs: AnyStr) -> Any:
            if hasattr(self, "skip_stages"):
                skip_stages = self.skip_stages
            else:
                raise ValueError("The class does not skip_stages")

            if stage_name in skip_stages:
                return skip_return
            else:
                return stage_fn(self, *args, **kwargs)

        return skip_stage_fn

    return skip_stage_decorator
