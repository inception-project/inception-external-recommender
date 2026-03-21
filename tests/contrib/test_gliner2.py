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
import pytest

from ariadne.contrib.gliner2 import Gliner2SpanRecommender
from ariadne.protocol import PredictionRequest
from tests.util import create_cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER
from ariadne.protocol import MTagset, MTag, MFeatureHandle


class DummyModel:
    def extract_entities(self, text: str, entity_types, include_spans=True, include_confidence=False):
        # Return a single entity span using the GLiNER2 extractor schema
        return {"entities": {"DUMMY": [{"start": 0, "end": 5}]}}


def test_gliner2_with_model_instance():
    cas = create_cas()
    cas.sofa_string = "Hello world"

    model = DummyModel()
    sut = Gliner2SpanRecommender(model)

    req = PredictionRequest(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc1", USER, tagsets=None)
    sut.on_prediction_request(req)

    predictions = list(cas.select(PREDICTED_TYPE))
    assert len(predictions) == 1
    pred = predictions[0]
    assert getattr(pred, PREDICTED_FEATURE) == "DUMMY"


class ScoredDummyModel:
    def extract_entities(self, text: str, entity_types, include_spans=True, include_confidence=True):
        return {"entities": {"DUMMY": [{"start": 0, "end": 5, "confidence": 0.42}]}}


def test_gliner2_scores_recorded():
    cas = create_cas()
    cas.sofa_string = "Hello world"

    model = ScoredDummyModel()
    sut = Gliner2SpanRecommender(model)

    req = PredictionRequest(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc1", USER, tagsets=None)
    sut.on_prediction_request(req)

    predictions = list(cas.select(PREDICTED_TYPE))
    assert len(predictions) == 1
    pred = predictions[0]
    score_name = f"{PREDICTED_FEATURE}_score"
    value = pred.get(score_name)
    assert value is not None
    assert float(value) == pytest.approx(0.42)


def test_gliner2_integration_local_model():
    """Integration test: load local GLiNER2 base model and run prediction."""
    from gliner2 import GLiNER2

    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    cas = create_cas()
    cas.sofa_string = "Apple CEO Tim Cook announced the iPhone in Cupertino."

    sut = Gliner2SpanRecommender(extractor)
    req = PredictionRequest(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_real", USER, tagsets=None)
    sut.on_prediction_request(req)

    preds = list(cas.select(PREDICTED_TYPE))
    assert len(preds) > 0


def test_gliner2_uses_tagset_schema():
    cas = create_cas()
    cas.sofa_string = "Aspirin 400mg"

    tag = MTag("MED", "Medication")
    target = MFeatureHandle(PREDICTED_TYPE, PREDICTED_FEATURE)
    ts = MTagset("MedTagset", tags=[tag], targets=[target])

    req = PredictionRequest(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc1", USER, tagsets=[ts])

    class CaptureModel:
        def __init__(self):
            self.last_schema = None

        def extract_entities(self, text: str, entity_types, include_spans=True, include_confidence=True):
            # capture the schema passed by the recommender
            self.last_schema = entity_types
            # return a single MED span
            return {"entities": {"MED": [{"start": 0, "end": 4}]}}

    model = CaptureModel()
    sut = Gliner2SpanRecommender(model)

    sut.on_prediction_request(req)

    # The model should have received a dict schema containing MED -> description
    assert isinstance(model.last_schema, dict)
    assert "med" in model.last_schema
    assert model.last_schema["med"] == "Medication"

    preds = list(cas.select(PREDICTED_TYPE))
    assert len(preds) == 1
    assert getattr(preds[0], PREDICTED_FEATURE) == "MED"
