from pathlib import Path

from ariadne.contrib import AdapterSequenceTagger
from ariadne.contrib.adapters import AdapterSentenceClassifier
from tests.util import (
    load_obama,
    PREDICTED_TYPE,
    PREDICTED_FEATURE,
    PROJECT_ID,
    USER,
    load_newsgroup_training_data,
    load_newsgroup_test_data,
)


def test_predict_pos(tmpdir_factory):
    cas = load_obama()
    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = AdapterSequenceTagger(
        base_model_name="bert-base-uncased",
        adapter_name="pos/ldc2012t13@vblagoje",
        labels=[
            "ADJ",
            "ADP",
            "ADV",
            "AUX",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SYM",
            "VERB",
            "X",
        ],
        model_directory=model_directory,
    )

    sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
    predictions = list(cas.select(PREDICTED_TYPE))

    assert len(predictions)

    for prediction in predictions:
        assert getattr(prediction, PREDICTED_FEATURE) is not None


def test_predict_sentiment(tmpdir_factory):
    training_data = load_newsgroup_training_data()
    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = AdapterSentenceClassifier(
        base_model_name="bert-base-uncased",
        adapter_name="sentiment/sst-2@ukp",
        labels=["negative", "positive"],
        model_directory=model_directory,
    )
    sut.fit(training_data, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER)

    test_data = load_newsgroup_test_data()

    for cas in test_data[:2]:
        sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
        # We have one sentence per document, therefore we expect exactly one prediction per document
        predictions = list(cas.select(PREDICTED_TYPE))
        prediction = predictions[0]
        assert prediction.value is not None
