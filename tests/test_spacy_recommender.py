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

pytest.importorskip("spacy")

from ariadne.contrib.spacy import SpacyPosClassifier, SpacyNerClassifier
from tests.util import load_obama, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER


def test_predict_ner(tmpdir_factory):
    cas = load_obama()
    sut = SpacyNerClassifier("en_core_web_sm")

    sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
    predictions = list(cas.select(PREDICTED_TYPE))

    assert len(predictions)

    for prediction in predictions:
        assert getattr(prediction, PREDICTED_FEATURE) is not None


def test_predict_pos(tmpdir_factory):
    cas = load_obama()
    sut = SpacyPosClassifier("en_core_web_sm")

    sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
    predictions = list(cas.select(PREDICTED_TYPE))

    assert len(predictions)

    for prediction in predictions:
        assert getattr(prediction, PREDICTED_FEATURE)
