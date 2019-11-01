import copy
from pathlib import Path
from typing import List

from cassis import TypeSystem, Cas

from sklearn.datasets import fetch_20newsgroups

from ariadne.constants import SENTENCE_TYPE, IS_PREDICTION
from ariadne.contrib import SklearnSentenceClassifier
from ariadne.protocol import TrainingDocument

_PREDICTED_TYPE = "ariadne.test.category"
_PREDICTED_FEATURE = "label"
_USER = "test_user"
_PROJECT_ID = "20newsgroups"
_CATEGORIES = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]


def test_fit(tmpdir_factory):
    training_data = _load_training_data()

    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = SklearnSentenceClassifier(model_directory)
    sut.fit(training_data, _PREDICTED_TYPE, _PREDICTED_FEATURE, _PROJECT_ID, _USER)

    model_path = model_directory / sut.name / f"model_{_USER}.joblib"
    assert model_path.is_file(), f"Expected {model_path} to be a file!"


def test_predict(tmpdir_factory):
    training_data = _load_training_data()
    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = SklearnSentenceClassifier(model_directory)
    sut.fit(training_data, _PREDICTED_TYPE, _PREDICTED_FEATURE, _PROJECT_ID, _USER)

    test_data = _load_test_data()

    for cas in test_data:
        sut.predict(cas, _PREDICTED_TYPE, _PREDICTED_FEATURE, _PROJECT_ID, "doc_42", _USER)
        # We have one sentence per document, therefore we expect exactly one prediction per document
        predictions = list(cas.select(_PREDICTED_TYPE))
        prediction = predictions[0]
        assert prediction.label is not None


def _load_training_data() -> List[TrainingDocument]:
    twenty_train = fetch_20newsgroups(subset="train", categories=_CATEGORIES, shuffle=True, random_state=42)
    target_names = twenty_train.target_names

    typesystem = _build_typesystem()
    SentenceType = typesystem.get_type(SENTENCE_TYPE)
    PredictedType = typesystem.get_type(_PREDICTED_TYPE)

    docs = []
    for i, (text, target) in enumerate(zip(twenty_train.data, twenty_train.target)):
        cas = Cas(typesystem=typesystem)
        cas.sofa_string = text

        begin = 0
        end = len(text)
        cas.add_annotation(SentenceType(begin=begin, end=end))
        cas.add_annotation(PredictedType(begin=begin, end=end, label=target_names[target]))

        doc = TrainingDocument(cas, f"doc_{i}", _USER)
        docs.append(doc)

    return docs


def _load_test_data() -> List[Cas]:
    twenty_test = fetch_20newsgroups(subset="test", categories=_CATEGORIES, shuffle=True, random_state=42)

    typesystem = _build_typesystem()
    SentenceType = typesystem.get_type(SENTENCE_TYPE)

    result = []
    for text in twenty_test.data[:5]:
        cas = Cas(typesystem=typesystem)
        cas.sofa_string = text

        begin = 0
        end = len(text)
        cas.add_annotation(SentenceType(begin=begin, end=end))

        result.append(cas)

    return result


def _build_typesystem() -> TypeSystem:
    typesystem = TypeSystem()
    SentenceType = typesystem.create_type(SENTENCE_TYPE)
    PredictedType = typesystem.create_type(_PREDICTED_TYPE)
    typesystem.add_feature(PredictedType, _PREDICTED_FEATURE, "uima.cas.String")
    typesystem.add_feature(PredictedType, IS_PREDICTION, "uima.cas.Boolean")
    return typesystem
