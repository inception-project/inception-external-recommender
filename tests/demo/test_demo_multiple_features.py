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

from ariadne.demo.demo_multiple_features import DemoMultipleFeaturesRecommender
from ariadne.protocol import TrainingDocument
from cassis import Cas
from ariadne.contrib.inception_util import create_span_prediction, TOKEN_TYPE
from tests.util import create_cas
from ariadne.contrib.inception_util import IS_PREDICTION


def test_demo_multiple_features_fit_and_predict():
    # Prepare a training CAS
    cas_train = create_cas()
    cas_train.sofa_string = "Hello world"
    # Create a custom predicted type that has two string features so the recommender
    # actually learns separate dictionaries per feature
    ts = cas_train.typesystem
    CustomPred = ts.create_type("ariadne.testtype_multi")
    ts.create_feature(CustomPred, "value1", "uima.cas.String")
    ts.create_feature(CustomPred, "value2", "uima.cas.String")
    ts.create_feature(CustomPred, IS_PREDICTION, "uima.cas.Boolean")

    # Add two training annotations on the same covered text but with different feature values
    span1 = create_span_prediction(cas_train, "ariadne.testtype_multi", "value1", 0, 5, "GREETING")
    span2 = create_span_prediction(cas_train, "ariadne.testtype_multi", "value2", 0, 5, "HELLO")
    cas_train.add(span1)
    cas_train.add(span2)

    docs = [TrainingDocument(cas_train, "doc1", "user1")]

    recommender = DemoMultipleFeaturesRecommender()
    # feature argument is ignored by the recommender
    recommender.fit(docs, "ariadne.testtype_multi", "value1", project_id=1, user_id="user1")

    # Create a new CAS to predict into using the same typesystem as the training CAS
    predict_cas = Cas(cas_train.typesystem)
    predict_cas.sofa_string = cas_train.sofa_string

    Token = predict_cas.typesystem.get_type(TOKEN_TYPE)
    predict_cas.add(Token(begin=0, end=5))
    predict_cas.add(Token(begin=6, end=11))

    recommender.predict(
        predict_cas,
        "ariadne.testtype_multi",
        "value1",
        project_id=1,
        document_id="doc1",
        user_id="user1",
    )

    # After prediction there should be predictions for both features
    preds_v1 = [a for a in predict_cas.select("ariadne.testtype_multi") if a.get("value1") == "GREETING"]
    preds_v2 = [a for a in predict_cas.select("ariadne.testtype_multi") if a.get("value2") == "HELLO"]

    assert len(preds_v1) >= 1
    assert len(preds_v2) >= 1
