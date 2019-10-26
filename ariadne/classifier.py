from typing import List, Iterator

from cassis import Cas
from cassis.typesystem import FeatureStructure, Type

from ariadne.constants import TOKEN_TYPE, IS_PREDICTION, SENTENCE_TYPE
from ariadne.protocol import TrainingDocument


class Classifier:
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id):
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
