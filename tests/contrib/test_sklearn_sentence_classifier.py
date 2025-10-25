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

pytest.importorskip("sklearn_crfsuite")

from pathlib import Path

from ariadne.contrib.sklearn import SklearnSentenceClassifier

from tests.util import *


def test_fit(tmpdir_factory):
    training_data = load_newsgroup_training_data()

    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = SklearnSentenceClassifier(model_directory)
    sut.fit(training_data, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER)

    model_path = model_directory / sut.name / f"model_{USER}.joblib"
    assert model_path.is_file(), f"Expected {model_path} to be a file!"


def test_predict(tmpdir_factory):
    training_data = load_newsgroup_training_data()
    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = SklearnSentenceClassifier(model_directory)
    sut.fit(training_data, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, USER)

    test_data = load_newsgroup_test_data()

    for cas in test_data:
        sut.predict(cas, PREDICTED_TYPE, PREDICTED_FEATURE, PROJECT_ID, "doc_42", USER)
        # We have one sentence per document, therefore we expect exactly one prediction per document
        predictions = list(cas.select(PREDICTED_TYPE))
        prediction = predictions[0]
        assert prediction.value is not None
