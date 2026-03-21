# Licensed to the Technische Universität Darmstadt under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The Technische Universität Darmstadt
# licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Any, List, Optional

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
    tagsets: Optional[List["MTagset"]] = attr.ib(default=None)


@attr.s
class MFeatureHandle:
    type: str = attr.ib()
    feature: str = attr.ib()


@attr.s
class MTag:
    tag: str = attr.ib()
    description: Optional[str] = attr.ib(default=None)


@attr.s
class MTagset:
    description: Optional[str] = attr.ib(default=None)
    tags: List[MTag] = attr.ib(factory=list)
    targets: List[MFeatureHandle] = attr.ib(factory=list)


@attr.s
class TrainingRequest:
    layer: str = attr.ib()
    feature: str = attr.ib()
    project_id: str = attr.ib()
    _typesystem_xml: str = attr.ib()
    _documents_json: List[Dict[str, str]] = attr.ib()

    @property
    def user_id(self) -> str:
        return self._documents_json[0]["userId"]

    @property
    def documents(self) -> List["TrainingDocument"]:
        # We parse this lazily as sometimes when already training, we just do not need to parse it at all.
        typesystem = load_typesystem(self._typesystem_xml)
        training_documents = []
        for document in self._documents_json:
            cas = load_cas_from_xmi(document["xmi"], typesystem)
            document_id = document["documentId"]
            user_id = document["userId"]
            training_documents.append(TrainingDocument(cas, document_id, user_id))

        return training_documents


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

    tagsets_json = json_object.get("tagsets")

    tagsets = None
    if tagsets_json is not None:
        parsed_tagsets: List[MTagset] = []
        for ts in tagsets_json:
            desc = ts.get("description")
            tags_json = ts.get("tags") or []
            targets_json = ts.get("targets") or []

            tags = [MTag(t.get("tag"), t.get("description")) for t in tags_json]
            targets = [MFeatureHandle(t.get("type"), t.get("feature")) for t in targets_json]

            parsed_tagsets.append(MTagset(desc, tags, targets))

        tagsets = parsed_tagsets

    return PredictionRequest(cas, layer, feature, project_id, document_id, user_id, tagsets)


def parse_training_request(json_object: JsonDict) -> TrainingRequest:
    metadata = json_object["metadata"]

    layer = metadata["layer"]
    feature = metadata["feature"]
    project_id = metadata["projectId"]
    typesystem_xml = json_object["typeSystem"]
    documents_json = json_object["documents"]

    return TrainingRequest(layer, feature, project_id, typesystem_xml, documents_json)
