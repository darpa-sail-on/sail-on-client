"""Activity Recognition Feedback."""

from sail_on_client.harness.local_harness import LocalHarness
from sail_on_client.harness.par_harness import ParHarness
from sail_on_client.feedback.feedback import Feedback
from typing import Union
import pandas as pd

SUPPORTED_FEEDBACK = ["classification", "score"]


class ActivityRecognitionFeedback(Feedback):
    """Feedback for activity recognition."""

    def __init__(
        self,
        first_budget: int,
        income_per_batch: int,
        maximum_budget: int,
        interface: Union[LocalHarness, ParHarness],
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
            feedback_type,
        )

    def get_labeled_feedback(
        self, round_id: int, images_id_list: list, image_names: list
    ) -> Union[pd.DataFrame, None]:
        """
        Get labeled feedback for the round.

        Args:
            round_id: Round identifier
            image_id_list: List if indices for images
            image_names: List of image names for the round

        Return:
            A dictionary with the accuracy value or None if
            feedback is requested for an older round
        """
        if round_id > self.current_round:
            self.deposit_income()
            self.current_round = round_id
            if len(images_id_list) <= self.budget:
                self.budget = self.budget - len(images_id_list)
                image_ids = [image_names[int(idx)] for idx in images_id_list]
                feedback_file = self.interface.get_feedback_request(
                    image_ids,
                    self.feedback_type,
                    self.test_id,
                    round_id,
                    self.session_id,
                )
                df = pd.read_csv(
                    feedback_file,
                    delimiter=",",
                    header=None,
                    names=["id", "class1", "class2", "class3", "class4", "class5"],
                )
            else:
                raise ValueError("the function should be added")
        else:
            df = None
        return df
