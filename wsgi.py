from inception_external_recommender.contrib.sklearn import SklearnSentenceClassifier
from inception_external_recommender.server import Server
from inception_external_recommender.util import setup_logging

if __name__ == '__main__':
    setup_logging()

    server = Server()
    # server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
    # server.add_classifier("spacy_pos", SpacyPosClassifier("en"))
    server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
    server.start()

