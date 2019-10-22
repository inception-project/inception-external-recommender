import logging
from typing import List, Optional

from cassis import Cas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline


from inception_external_recommender.classifier import Classifier
from inception_external_recommender.modelmanager import ModelManager
from inception_external_recommender.protocol import TrainingDocument


logger = logging.getLogger(__file__)

_CLASSIFIER_NAME = "sklearn-sentence-classifier"


class SklearnSentenceClassifier(Classifier):

    def __init__(self):
        self._model_manager = ModelManager("")

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id):
        user_id = documents[0].user_id

        with self._model_manager.lock_model(_CLASSIFIER_NAME, user_id):
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

            self._model_manager.save_model(_CLASSIFIER_NAME, user_id, model)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model: Optional[Pipeline] = self._model_manager.load_model(_CLASSIFIER_NAME, user_id)

        if model is None:
            logger.debug("No trained model ready yet!")
            return

        for sentence in self.iter_sentences(cas):
            predicted = model.predict([cas.get_covered_text(sentence)])[0]
            prediction = self.create_prediction(cas, layer, feature, sentence.begin, sentence.end, predicted)
            cas.add_annotation(prediction)
