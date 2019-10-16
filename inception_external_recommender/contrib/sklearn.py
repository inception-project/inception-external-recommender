import logging
from typing import List

from cassis import Cas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline


from inception_external_recommender.classifier import Classifier
from inception_external_recommender.protocol import TrainingDocument


logger = logging.getLogger(__file__)


class SklearnSentenceClassifier(Classifier):

    def __init__(self):
        self._model = None

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id):
        sentences = []
        targets = []

        for document in documents:
            cas = document.cas

            for sentence in self.iter_sentences(cas):
                try:
                    annotation = next(cas.select_covered(layer, sentence))
                except StopIteration:
                    continue

                label = getattr(annotation, feature)

                sentences.append(cas.get_covered_text(sentence))
                targets.append(label)

        logging.debug(f"Training on {len(sentences)} sentences")

        model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
        model.fit(sentences, targets)
        self._model = model

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        if self._model is None:
            logger.debug("No trained model ready yet!")
            return

        for sentence in self.iter_sentences(cas):
            predicted = self._model.predict([cas.get_covered_text(sentence)])[0]
            prediction = self.create_prediction(cas, layer, feature, sentence.begin, sentence.end, predicted)
            cas.add_annotation(prediction)
