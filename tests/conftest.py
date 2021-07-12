"""Configuration used by pytest to register fixtures."""
pytest_plugins = [
    "tests.helpers",
    "tests.activity_recognition_metric_values",
    "tests.transcription_metric_values",
    "tests.image_classification_metric_values",
]
