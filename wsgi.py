from ariadne.contrib.jieba import JiebaSegmenter
from ariadne.contrib.nltk import NltkStemmer
from ariadne.contrib.sbert import SbertSentenceClassifier
from ariadne.contrib.sklearn import SklearnSentenceClassifier, SklearnMentionDetector
from ariadne.contrib.stringmatcher import LevenshteinStringMatcher
from ariadne.server import Server
from ariadne.util import setup_logging

setup_logging()

server = Server()
# server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
# server.add_classifier("spacy_pos", SpacyPosClassifier("en"))
# server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
#server.add_classifier("jieba", JiebaSegmenter())
#server.add_classifier("stemmer", NltkStemmer())
#server.add_classifier("leven", LevenshteinStringMatcher())
server.add_classifier("sbert", SbertSentenceClassifier())

server.start(debug=True, port=40022)
