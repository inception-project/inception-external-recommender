from typing import List

from cassis import Cas
from cassis.typesystem import FeatureStructure, Type

from inception_external_recommender.constants import TOKEN_TYPE, IS_PREDICTION


class Classifier:

    def fit(self):
        raise NotImplementedError()

    def predict(self, document: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        raise NotImplementedError()

    def get_tokens(self, document: Cas) -> List[FeatureStructure]:
        """ Returns the token texts of all tokens in the given document.

        Args:
            document:

        Returns:

        """
        tokens = list(document.select(TOKEN_TYPE))
        return tokens

    def create_prediction(self, document: Cas, layer: str, feature: str, begin: int, end: int, label: str) -> FeatureStructure:
        AnnotationType = document.typesystem.get_type(layer)

        fields = {'begin': begin,
                  'end': end,
                  IS_PREDICTION: True,
                  feature: label}
        prediction = AnnotationType(**fields)
        return prediction
