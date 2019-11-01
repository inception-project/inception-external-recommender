import logging
import os
from pathlib import Path
from typing import List, Iterator, Optional, Any

import joblib
from cassis import Cas
from cassis.typesystem import FeatureStructure, Type

import ariadne
from ariadne.constants import TOKEN_TYPE, IS_PREDICTION, SENTENCE_TYPE
from ariadne.protocol import TrainingDocument

logger = logging.getLogger(__file__)


class Classifier:
    def __init__(self, model_directory: Path = None):
        self.model_directory = ariadne.model_directory if model_directory is None else model_directory

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        pass

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        raise NotImplementedError()

    def iter_sentences(self, cas: Cas) -> Iterator[FeatureStructure]:
        """ Returns an iterator over all sentences in the given document.

        Args:
            cas:

        Returns:

        """
        return cas.select(SENTENCE_TYPE)

    def iter_tokens(self, cas: Cas) -> Iterator[FeatureStructure]:
        """ Returns an iterator over all tokens in the given document.

        Args:
            cas:

        Returns:

        """
        return cas.select(TOKEN_TYPE)

    def get_tokens(self, cas: Cas) -> List[FeatureStructure]:
        """ Returns the token of all tokens in the given document.

        Args:
            cas:

        Returns:

        """
        return list(self.iter_tokens(cas))

    def create_prediction(
        self, cas: Cas, layer: str, feature: str, begin: int, end: int, label: str
    ) -> FeatureStructure:
        AnnotationType = cas.typesystem.get_type(layer)

        fields = {"begin": begin, "end": end, IS_PREDICTION: True, feature: label}
        prediction = AnnotationType(**fields)
        return prediction

    def _load_model(self, user_id: str) -> Optional[Any]:
        model_path = self._get_model_path(user_id)
        if model_path.is_file():
            logger.debug("Model found for [%s]", model_path)
            return joblib.load(model_path)
        else:
            logger.debug("No model found for [%s]", model_path)
            return None

    def _save_model(self, user_id: str, model: Any):
        model_path = self._get_model_path(user_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_model_path = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, tmp_model_path)
        os.replace(tmp_model_path, model_path)

    def _get_model_path(self, user_id: str) -> Path:
        return self.model_directory / self.name / f"model_{user_id}.joblib"

    @property
    def name(self) -> str:
        return type(self).__name__
