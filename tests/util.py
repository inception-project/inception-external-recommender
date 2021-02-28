from typing import List

from cassis import *

from sklearn.datasets import fetch_20newsgroups

from ariadne.contrib.inception_util import SENTENCE_TYPE, IS_PREDICTION
from ariadne.protocol import TrainingDocument

PREDICTED_TYPE = "ariadne.testtype"
PREDICTED_FEATURE = "value"
USER = "test_user"
PROJECT_ID = "test_project"
NEWSGROUP_CATEGORIES = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]


def load_newsgroup_training_data() -> List[TrainingDocument]:
    twenty_train = fetch_20newsgroups(subset="train", categories=NEWSGROUP_CATEGORIES, shuffle=True, random_state=42)
    target_names = twenty_train.target_names

    typesystem = build_typesystem()
    SentenceType = typesystem.get_type(SENTENCE_TYPE)
    PredictedType = typesystem.get_type(PREDICTED_TYPE)

    docs = []
    for i, (text, target) in enumerate(zip(twenty_train.data, twenty_train.target)):
        cas = Cas(typesystem=typesystem)
        cas.sofa_string = text

        begin = 0
        end = len(text)
        cas.add_annotation(SentenceType(begin=begin, end=end))
        cas.add_annotation(PredictedType(begin=begin, end=end, value=target_names[target]))

        doc = TrainingDocument(cas, f"doc_{i}", USER)
        docs.append(doc)

    return docs


def load_newsgroup_test_data() -> List[Cas]:
    twenty_test = fetch_20newsgroups(subset="test", categories=NEWSGROUP_CATEGORIES, shuffle=True, random_state=42)

    typesystem = build_typesystem()
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


def load_obama() -> Cas:
    with open("data/INCEpTION_TypeSystem.xml", "rb") as f:
        typesystem = merge_typesystems(load_typesystem(f), build_typesystem())

    with open("data/Wikipedia-Obama.xmi", "rb") as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)

    return cas


def build_typesystem() -> TypeSystem:
    typesystem = TypeSystem()
    SentenceType = typesystem.create_type(SENTENCE_TYPE)
    PredictedType = typesystem.create_type(PREDICTED_TYPE)
    typesystem.add_feature(PredictedType, PREDICTED_FEATURE, "uima.cas.String")
    typesystem.add_feature(PredictedType, IS_PREDICTION, "uima.cas.Boolean")
    return typesystem
