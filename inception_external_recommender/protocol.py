from collections import namedtuple

from typing import Dict, Any

from cassis import load_cas_from_xmi, load_typesystem

# Types

JsonDict = Dict[str, Any]

# Data classes

PredictionRequest = namedtuple("PredictionRequest", ["document", "layer", "feature", "project_id", "document_id", "user_id"])
PredictionResponse = namedtuple("PredictionResponse", ["document"])
Document = namedtuple("Document", ["cas", "documentId", "userId"])


def parse_prediction_request(json_object: JsonDict) -> PredictionRequest:
    metadata = json_object["metadata"]
    document = json_object["document"]

    layer = metadata["layer"]
    feature = metadata["feature"]
    project_id = metadata["projectId"]

    typesystem = load_typesystem(json_object["typeSystem"])
    cas = load_cas_from_xmi(document["xmi"], typesystem)
    document_id = document["documentId"]
    user_id = document["userId"]

    return PredictionRequest(cas, layer, feature, project_id,  document_id, user_id)
