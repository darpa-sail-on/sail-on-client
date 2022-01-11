"""Document Transcription Feedback."""

import pandas as pd
from sail_on_client.harness.local_harness import LocalHarness
from sail_on_client.harness.par_harness import ParHarness
from sail_on_client.feedback.feedback import Feedback

from typing import Union, Dict

SUPPORTED_FEEDBACK = ["classification", "score", "transcription"]


class DocumentTranscriptionFeedback(Feedback):
    """Feedback for document transcription."""

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
        Initialize document transcription feedback object.

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
        super(DocumentTranscriptionFeedback, self).__init__(
            first_budget,
            income_per_batch,
            maximum_budget,
            interface,
            session_id,
            test_id,
            feedback_type,
        )
        self.current_round: int = -1
        self.budget: int = first_budget

    def get_levenshtein_feedback(
        self, round_id: int, images_id_list: list, image_names: list
    ) -> Union[Dict, None]:
        """
        Get levenshtein feedback for the round.

        Args:
            round_id: Round identifier
            image_id_list: List if indices for images
            image_names: List of image names for the round

        Return:
            A dictionary containing levenshtein score or None if
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
                df = pd.read_csv(feedback_file, delimiter=",", header=None)
                return df
            else:
                raise ValueError("the function should be added")
        else:
            return None

    def get_feedback(
        self, round_id: int, images_id_list: list, image_names: list
    ) -> Union[pd.DataFrame, Dict, None]:
        """
        Get feedback for the round.

        Args:
            round_id: Round identifier
            image_id_list: List if indices for images
            image_names: List of image names for the round

        Return:
            Either a dataframe or dictionary with score if the request is valid
            for the current round.
        """
        if self.feedback_type == "classification":
            feedback_fn = self.get_labeled_feedback
        elif self.feedback_type == "score":
            feedback_fn = self.get_score_feedback
        elif self.feedback_type == "transcription":
            feedback_fn = self.get_levenshtein_feedback
        else:
            raise ValueError("Unsupported feedback type {self.feedback_type} specified")
        return feedback_fn(round_id, images_id_list, image_names)
