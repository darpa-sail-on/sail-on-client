"""Exception objects representing HTTP error conditions."""

from typing import Any, List, Tuple, Dict


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
    """Base class for all server side error conditions."""

    reason = "Unknown"
    msg = ""
    stack_trace = "stack trace unavailable"
    error_code = 500

    def __init__(
        self, reason: str, msg: str, stack_trace: str = "stack trace unavailable"
    ):
        """
        Initialize the api error object.

        Args:
            reason: Cause for the error
            msg: Additional message associated with the error
            stack_trace: Stack trace associated with the error

        Returns:
            None
        """
        self.reason = reason
        self.msg = msg
        self.stack_trace = stack_trace

    @staticmethod
    def error_classes() -> List[Any]:
        """
        Return all error classes in the system.

        Returns:
            List of subclasses of server error
        """
        return _get_all_subclasses(ApiError)

    def flask_response(self) -> Tuple[Dict[str, str], int]:
        """
        Convert the object to a Flask response.

        Returns:
            Tuple of response obtained from server and error code
        """
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
        """
        Initialize the server error object.

        Args:
            reason: Cause for the error
            msg: Additional message associated with the error
            stack_trace: Stack trace associated with the error

        Returns:
            None
        """
        super().__init__(reason, msg, stack_trace)


class RoundError(ServerError):
    """Error indicating problem with rounds."""

    error_code = 204

    def __init__(
        self, reason: str, msg: str, stack_trace: str = "stack trace unavailable"
    ):
        """
        Initialize the round error object.

        Args:
            reason: Cause for the error
            msg: Additional message associated with the error
            stack_trace: Stack trace associated with the error

        Returns:
            None
        """
        super().__init__(reason, msg, stack_trace)


class ProtocolError(ApiError):
    """400-series error indicating caller caused a problem engaging a protocol."""

    error_code = 400

    def __init__(
        self, reason: str, msg: str, stack_trace: str = "stack trace unavailable"
    ):
        """
        Initialize the Protocol error object.

        Args:
            reason: Cause for the error
            msg: Additional message associated with the error
            stack_trace: Stack trace associated with the error

        Returns:
            None
        """
        super().__init__(reason, msg, stack_trace)
