from cassis import Cas
from cassis.typesystem import FeatureStructure

SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
IS_PREDICTION = "inception_internal_predicted"


def create_prediction(cas: Cas, layer: str, feature: str, begin: int, end: int, label: str) -> FeatureStructure:
    AnnotationType = cas.typesystem.get_type(layer)

    fields = {"begin": begin, "end": end, IS_PREDICTION: True, feature: label}
    prediction = AnnotationType(**fields)
    return prediction
