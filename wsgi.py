from ariadne.contrib.sklearn import SklearnSentenceClassifier
from ariadne.server import Server
from ariadne.util import setup_logging

setup_logging()

server = Server()
# server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
# server.add_classifier("spacy_pos", SpacyPosClassifier("en"))
server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
