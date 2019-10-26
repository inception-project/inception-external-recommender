import logging
import time
from typing import List, Optional

from cassis import Cas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline


from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument


logger = logging.getLogger(__name__)


class SklearnSentenceClassifier(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.debug("Start training for user [%s]", user_id)
        sentences = []
        targets = []

        for document in documents:
            cas = document.cas

            for sentence in self.iter_sentences(cas):
                # Get the first annotation that covers the sentence
                try:
                    annotation = next(cas.select_covered(layer, sentence))
                except StopIteration:
                    continue

                assert (
                    sentence.begin == annotation.begin and sentence.end == annotation.end
                ), "Annotation should cover sentence fully!"

                label = getattr(annotation, feature)

                sentences.append(cas.get_covered_text(sentence))
                targets.append(label)

        logger.debug(f"Training on {len(sentences)} sentences")
        time.sleep(10)

        model = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", MultinomialNB())])
        model.fit(sentences, targets)
        time.sleep(10)
        logger.debug(f"Training finished")

        self._save_model(user_id, model)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model: Optional[Pipeline] = self._load_model(user_id)

        if model is None:
            logger.debug("No trained model ready yet!")
            return

        for sentence in self.iter_sentences(cas):
            predicted = model.predict([cas.get_covered_text(sentence)])[0]
            prediction = self.create_prediction(cas, layer, feature, sentence.begin, sentence.end, predicted)
            cas.add_annotation(prediction)
