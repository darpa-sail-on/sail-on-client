"""Exception objects representing HTTP error conditions."""

from typing import Any, List, Tuple, TypeVar, Dict


def _get_all_subclasses(cls: Any) -> List[Any]:
    from itertools import chain

    return list(
        chain.from_iterable(
            [
                list(chain.from_iterable([[x], _get_all_subclasses(x)]))
                for x in cls.__subclasses__()
            ]
        )
    )


class ApiError(Exception):
    """Base class for all serverside error conditions."""

    reason = "Unknown"
    msg = ""
    stack_trace = "stack trace unavailable"
    error_code = 500

    def __init__(
        self, reason: str, msg: str, stack_trace: str = "stack trace unavailable"
    ):
        """Initialize the error object."""
        self.reason = reason
        self.msg = msg
        self.stack_trace = stack_trace

    @staticmethod
    def error_classes() -> List[Any]:
        """Return all error classes in the system."""
        return _get_all_subclasses(ApiError)

    def flask_response(self) -> Tuple[Dict[str, str], int]:
        """Convert the object to a Flask response."""
        response = {
            "reason": self.reason,
            "message": self.msg,
            "stack_trace": self.stack_trace,
        }

        return response, self.error_code


class ServerError(ApiError):
    """500-series error indicating general problem on the serverside."""

    error_code = 500

    def __init__(
        self, reason: str, msg: str, stack_trace: str = "stack trace unavailable"
    ):
        """Initialize."""
        super().__init__(reason, msg, stack_trace)


class RoundError(ServerError):
    """Error indicating problem with rounds."""

    error_code = 502

    def __init__(
        self, reason: str, msg: str, stack_trace: str = "stack trace unavailable"
    ):
        """Initialize."""
        super().__init__(reason, msg, stack_trace)


class ProtocolError(ApiError):
    """400-series error indicating caller caused a problem engaging a protocol."""

    error_code = 400

    def __init__(
        self, reason: str, msg: str, stack_trace: str = "stack trace unavailable"
    ):
        """Initialize the error object."""
        super().__init__(reason, msg, stack_trace)
