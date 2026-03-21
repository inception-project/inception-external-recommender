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
from types import SimpleNamespace

from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument


class DummyClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.fit_called = False
        self.predict_called = False
        self.fit_args = None
        self.predict_args = None

    def fit(self, documents, layer, feature, project_id, user_id: str):
        self.fit_called = True
        self.fit_args = (documents, layer, feature, project_id, user_id)

    def predict(self, cas, layer, feature, project_id, document_id, user_id: str):
        self.predict_called = True
        self.predict_args = (cas, layer, feature, project_id, document_id, user_id)


def test_on_training_request_calls_fit():
    clf = DummyClassifier()

    # Create a fake TrainingRequest-like object with a prebuilt documents list
    td = TrainingDocument(None, "doc1", "u1")
    req = SimpleNamespace(layer="L", feature="F", project_id="P", user_id="u1", documents=[td])

    clf.on_training_request(req)  # should call fit

    assert clf.fit_called
    assert clf.fit_args[1:] == ("L", "F", "P", "u1")


def test_on_prediction_request_calls_predict():
    clf = DummyClassifier()

    req = SimpleNamespace(cas=None, layer="L", feature="F", project_id="P", document_id="D", user_id="u2")

    clf.on_prediction_request(req)  # should call predict

    assert clf.predict_called
    assert clf.predict_args[1:] == ("L", "F", "P", "D", "u2")


def test_override_on_training_request_short_circuits():
    class ShortCircuitClassifier(DummyClassifier):
        def on_training_request(self, req):
            # do not call super().on_training_request -> fit should not be called
            self.short = True

    clf = ShortCircuitClassifier()
    td = TrainingDocument(None, "doc1", "u1")
    req = SimpleNamespace(layer="L", feature="F", project_id="P", user_id="u1", documents=[td])

    clf.on_training_request(req)
    assert hasattr(clf, "short") and clf.short is True
    assert not clf.fit_called
