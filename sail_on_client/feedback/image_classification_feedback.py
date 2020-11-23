"""Image Classification Feedback."""
import pandas as pd

from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface

from typing import Union


class ImageClassificationFeedback:
    """Feedback for image classification."""

    def __init__(
        self,
        first_budget: int,
        income_per_batch: int,
        maximum_budget: int,
        interface: Union[LocalInterface, ParInterface],
        session_id: str,
        test_id: str,
        feedback_type: str = "classification",
    ) -> None:
        """Initialize."""
        self.budget = first_budget
        self.income_per_batch = income_per_batch
        self.maximum_budget = maximum_budget
        self.current_round = -1
        self.interface = interface
        self.session_id = session_id
        self.test_id = test_id
        self.feedback_type = feedback_type

    def get_feedback(
        self, round_id: int, images_id_list: list, image_names: list
    ) -> Union[pd.DataFrame, None]:
        """Get feedback for the round."""
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
                    feedback_file, delimiter=",", header=None, names=["id", "labels"]
                )
            else:
                raise ValueError("the function should be added")
        else:
            df = None
        return df

    def deposit_income(self) -> None:
        """Get income for a round."""
        self.budget = min(self.maximum_budget, (self.budget + self.income_per_batch))

    def get_budget(self) -> int:
        """Get current budget."""
        return self.budget
