from pathlib import Path

import pytest

from ariadne.contrib import SbertSentenceClassifier

from tests.util import *


def test_fit(tmpdir_factory):
    training_data = load_newsgroup_training_data()

    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = SbertSentenceClassifier(model_directory)
    sut.fit(training_data, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER)

    model_path = model_directory / sut.name / f"model_{USER}.joblib"
    assert model_path.is_file(), f"Expected {model_path} to be a file!"


def test_predict(tmpdir_factory):
    training_data = load_newsgroup_training_data()
    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = SbertSentenceClassifier(model_directory)
    sut.fit(training_data, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER)

    test_data = load_newsgroup_test_data()[:2]

    for cas in test_data:
        sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
        # We have one sentence per document, therefore we expect exactly one prediction per document
        predictions = list(cas.select(PREDICTED_TYPE))
        prediction = predictions[0]
        assert prediction.value is not None
