from ariadne.contrib.spacy import SpacyPosClassifier, SpacyNerClassifier
from tests.util import load_obama, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER


def test_predict_ner(tmpdir_factory):
    cas = load_obama()
    sut = SpacyNerClassifier("en_core_web_sm")

    sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
    predictions = list(cas.select(PREDICTED_TYPE))

    assert len(predictions)

    for prediction in predictions:
        assert getattr(prediction, PREDICTED_FEATURE) is not None


def test_predict_pos(tmpdir_factory):
    cas = load_obama()
    sut = SpacyPosClassifier("en_core_web_sm")

    sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
    predictions = list(cas.select(PREDICTED_TYPE))

    assert len(predictions)

    for prediction in predictions:
        assert getattr(prediction, PREDICTED_FEATURE)
