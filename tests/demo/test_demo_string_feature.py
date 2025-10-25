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
from ariadne.demo.demo_string_feature import DemoStringFeatureRecommender
from ariadne.protocol import TrainingDocument
from ariadne.contrib.inception_util import create_span_prediction, TOKEN_TYPE
from tests.util import create_cas, PREDICTED_TYPE, PREDICTED_FEATURE


def test_demo_string_feature_fit_and_predict():
    # Prepare a training CAS and add a PREDICTED_TYPE annotation that contains the label
    cas_train = create_cas()
    cas_train.sofa_string = "Hello world"

    # Use the test predicted type (defined in tests.util) which already contains
    # the prediction flag feature expected by create_span_prediction
    span = create_span_prediction(cas_train, PREDICTED_TYPE, PREDICTED_FEATURE, 0, 5, "GREETING")
    cas_train.add(span)

    # Train the recommender on the test predicted type
    docs = [TrainingDocument(cas_train, "doc1", "user1")]

    recommender = DemoStringFeatureRecommender()
    recommender.fit(docs, PREDICTED_TYPE, PREDICTED_FEATURE, project_id=1, user_id="user1")

    # Create a new CAS to predict into and add a Token annotation for 'Hello'
    predict_cas = create_cas()
    predict_cas.sofa_string = cas_train.sofa_string

    # Add Token annotations so predict() will iterate over them for both words
    Token = predict_cas.typesystem.get_type(TOKEN_TYPE)
    predict_cas.add(Token(begin=0, end=5))
    predict_cas.add(Token(begin=6, end=11))

    # Run prediction; predictions will be created as instances of PREDICTED_TYPE
    recommender.predict(
        predict_cas,
        PREDICTED_TYPE,
        PREDICTED_FEATURE,
        project_id=1,
        document_id="doc1",
        user_id="user1",
    )

    # After prediction, there should be at least one PREDICTED_TYPE annotation with value 'GREETING'
    preds = [a for a in predict_cas.select(PREDICTED_TYPE) if a.get(PREDICTED_FEATURE) == "GREETING"]
    assert len(preds) >= 1
