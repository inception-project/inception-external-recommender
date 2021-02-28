import logging
from typing import List

from cassis import Cas
from diskcache import Cache
from sentence_transformers import SentenceTransformer

from lightgbm import LGBMClassifier


import numpy as np

from ariadne import cache_directory
from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE
from ariadne.protocol import TrainingDocument


logger = logging.getLogger(__name__)


class CachedSentenceTransformer:
    def __init__(self, model_name: str):
        super().__init__()
        self._model = SentenceTransformer(model_name)
        self._cache = Cache(cache_directory / model_name)

    def featurize(self, sentences: List[str]) -> np.ndarray:
        result = []
        for sentence in sentences:
            if sentence in self._cache:
                vec = self._cache[sentence]
            else:
                vec = self._model.encode(sentence).squeeze()
                self._cache[sentence] = vec

            result.append(vec)

        return np.array(result)

    def get_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class SbertSentenceClassifier(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.debug("Start training for user [%s]", user_id)

        featurizer = self._get_featurizer()

        sentences = []
        targets = []

        for document in documents:
            cas = document.cas

            for sentence in cas.select(SENTENCE_TYPE):
                # Get the first annotation that covers the sentence
                annotations = cas.select_covered(layer, sentence)

                if len(annotations):
                    annotation = annotations[0]
                else:
                    continue

                assert (
                    sentence.begin == annotation.begin and sentence.end == annotation.end
                ), "Annotation should cover sentence fully!"

                label = getattr(annotation, feature)

                if label is None:
                    continue

                sentences.append(cas.get_covered_text(sentence))
                targets.append(label)

        featurized_sentences = featurizer.featurize(sentences)

        logger.debug(f"Training started for user [%s]", user_id)
        model = LGBMClassifier().fit(featurized_sentences, targets)
        logger.debug(f"Training finished for user [%s]", user_id)

        self._save_model(user_id, model)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model = self._load_model(user_id)

        if model is None:
            logger.debug("No trained model ready yet!")
            return

        featurizer = self._get_featurizer()
        sentences = cas.select(SENTENCE_TYPE)
        featurized_sentences = featurizer.featurize([s.get_covered_text() for s in sentences])
        predictions = model.predict(featurized_sentences)

        for sentence, featurized_sentence, label in zip(sentences, featurized_sentences, predictions):
            prediction = create_prediction(cas, layer, feature, sentence.begin, sentence.end, label)
            cas.add_annotation(prediction)

    def _get_featurizer(self):
        return CachedSentenceTransformer("distilbert-base-nli-mean-tokens")
