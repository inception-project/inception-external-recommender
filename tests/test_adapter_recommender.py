from pathlib import Path

from ariadne.contrib import AdapterSequenceTagger
from tests.util import load_obama, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER


def test_predict(tmpdir_factory):
    cas = load_obama()
    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = AdapterSequenceTagger(model_directory)

    sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
    predictions = list(cas.select(PREDICTED_TYPE))

    assert len(predictions)

    for prediction in predictions:
        assert getattr(prediction, PREDICTED_FEATURE) is not None
