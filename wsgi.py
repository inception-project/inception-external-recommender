from ariadne.contrib import *
from ariadne.server import Server
from ariadne.util import setup_logging

setup_logging()

server = Server()
# server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
# server.add_classifier("spacy_pos", SpacyPosClassifier("en"))
# server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
# server.add_classifier("jieba", JiebaSegmenter())
# server.add_classifier("stemmer", NltkStemmer())
# server.add_classifier("leven", LevenshteinStringMatcher())
# server.add_classifier("sbert", SbertSentenceClassifier())
server.add_classifier("adapterpos", AdapterSequenceTagger())

server.start(debug=True, port=40022)
