"""Tests for create_metric_instance."""

from sail_on_client.evaluate import create_metric_instance
from sail_on_client.evaluate.activity_recognition import ActivityRecognitionMetrics
from sail_on_client.evaluate.image_classification import ImageClassificationMetrics
from sail_on_client.evaluate.document_transcription import DocumentTranscriptionMetrics
import pytest

ic_dict = {
    "image_id": 0,
    "detection": 1,
    "classification": 2
}

dt_dict = {
    "image_id": 0,
    "text": 1,
    "novel": 2,
    "representation": 3,
    "detection": 4,
    "classification": 5,
    "pen_pressure": 6,
    "letter_size": 7,
    "word_spacing": 8,
    "slant_angle": 9,
    "attribute": 10
}

ar_dict = {
    "video_id": 0,
    "novel": 1,
    "detection": 2,
    "classification": 3,
    "spatial": 4,
    "temporal": 5
}


@pytest.mark.parametrize(
    "protocol", ["OND", "CONDDA"]
)
@pytest.mark.parametrize(
    "domain,gt_dict,expected", [("image_classification", ic_dict, ImageClassificationMetrics),
                                ("transcripts", dt_dict, DocumentTranscriptionMetrics),
                                ("activity_recognition", ar_dict, ActivityRecognitionMetrics)]
)
def test_create_metric_instance(protocol, domain, gt_dict, expected):
    """
    Test for creating metric instance.

    Args:
        protocol: Name of the protocol
        domain: Name of the domain
        gt_dict: Parameters for the class created by metric
        expected: Expected Output Class

    Returns:
        None
    """
    metric = create_metric_instance(protocol, domain, gt_dict)
    assert isinstance(metric, expected)
