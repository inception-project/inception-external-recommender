from typing import Dict, Any, List

import attr
import cassis

from cassis import load_cas_from_xmi, load_typesystem

# Types

JsonDict = Dict[str, Any]

# Data classes


@attr.s
class PredictionRequest:
    cas: cassis.Cas = attr.ib()
    layer: str = attr.ib()
    feature: str = attr.ib()
    project_id: str = attr.ib()
    document_id: str = attr.ib()
    user_id: str = attr.ib()


@attr.s
class TrainingRequest:
    documents: List["TrainingDocument"] = attr.ib()
    layer: str = attr.ib()
    feature: str = attr.ib()
    project_id: str = attr.ib()


@attr.s
class TrainingDocument:
    cas: cassis.Cas = attr.ib()
    document_id: str = attr.ib()
    user_id: str = attr.ib()


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


def parse_training_request(json_object: JsonDict) -> TrainingRequest:
    metadata = json_object["metadata"]

    layer = metadata["layer"]
    feature = metadata["feature"]
    project_id = metadata["projectId"]

    typesystem = load_typesystem(json_object["typeSystem"])
    training_documents = []
    for document in json_object["documents"]:
        cas = load_cas_from_xmi(document["xmi"], typesystem)
        document_id = document["documentId"]
        user_id = document["userId"]
        training_documents.append(TrainingDocument(cas, document_id, user_id))

    return TrainingRequest(training_documents, layer, feature, project_id)
