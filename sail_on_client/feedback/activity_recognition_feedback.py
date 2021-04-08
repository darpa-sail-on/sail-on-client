"""Activity Recognition Feedback."""

from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from sail_on_client.feedback.feedback import Feedback
from typing import Union

SUPPORTED_FEEDBACK = ["classification", "score"]


class ActivityRecognitionFeedback(Feedback):
    """Feedback for activity recognition."""

    def __init__(
        self,
        first_budget: int,
        income_per_batch: int,
        maximum_budget: int,
        interface: Union[LocalInterface, ParInterface],
        session_id: str,
        test_id: str,
        feedback_type: str,
    ) -> None:
        """
        Initialize activity recognition feedback object.

        Args:
            first_budget: Initial budget
            income_per_batch: Additional labels added after every batch
            maximum_budget: Max labels that can be requested
            interface: An instance of evaluation interface
            session_id: Session identifier
            test_id: Test identifier
            feedback_type: Type of feedback that can be requested

        Returns:
            None
        """
        if feedback_type not in SUPPORTED_FEEDBACK:
            raise ValueError(f"Unsupported feedback_type {feedback_type}")
        super(ActivityRecognitionFeedback, self).__init__(
            first_budget,
            income_per_batch,
            maximum_budget,
            interface,
            session_id,
            test_id,
            feedback_type
        )
