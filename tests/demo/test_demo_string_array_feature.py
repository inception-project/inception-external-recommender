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
from cassis import Cas
from cassis.typesystem import TYPE_NAME_STRING_ARRAY, TYPE_NAME_BOOLEAN
from ariadne.contrib.inception_util import IS_PREDICTION

from ariadne.demo.demo_string_array_feature import DemoStringArrayFeatureRecommender
from ariadne.protocol import TrainingDocument
from ariadne.contrib.inception_util import create_span_prediction, TOKEN_TYPE
from tests.util import create_cas

PRED_TYPE = "ariadne.testtype_array"
PRED_FEATURE = "values"


def test_demo_string_array_feature_fit_and_predict():
    # Build a CAS with a predicted-type that has a string array feature
    cas_train = create_cas()
    cas_train.sofa_string = "Hello world"

    # Create a predicted type with a StringArray feature on the training CAS
    ts = cas_train.typesystem
    ArrayPredType = ts.create_type(PRED_TYPE)
    ts.create_feature(ArrayPredType, PRED_FEATURE, TYPE_NAME_STRING_ARRAY)
    ts.create_feature(ArrayPredType, IS_PREDICTION, TYPE_NAME_BOOLEAN)

    # Create a StringArray instance and attach it to an annotation
    StringArray = cas_train.typesystem.get_type(TYPE_NAME_STRING_ARRAY)
    labels_inst = StringArray(elements=["GREETING"])
    span = create_span_prediction(cas_train, PRED_TYPE, PRED_FEATURE, 0, 5, labels_inst)
    cas_train.add(span)

    recommender = DemoStringArrayFeatureRecommender()

    docs = [TrainingDocument(cas_train, "doc1", "user1")]
    recommender.fit(docs, PRED_TYPE, PRED_FEATURE, project_id=1, user_id="user1")

    # Create predict cas and add tokens for both words; ensure same predicted type exists
    predict_cas = Cas(cas_train.typesystem)
    predict_cas.sofa_string = cas_train.sofa_string

    Token = predict_cas.typesystem.get_type(TOKEN_TYPE)
    predict_cas.add(Token(begin=0, end=5))
    predict_cas.add(Token(begin=6, end=11))

    recommender.predict(predict_cas, PRED_TYPE, PRED_FEATURE, project_id=1, document_id="doc1", user_id="user1")

    # After prediction, there should be a predicted type with the feature containing the expected label
    preds = [a for a in predict_cas.select(PRED_TYPE) if getattr(a, PRED_FEATURE, None) is not None]
    assert len(preds) == 1
    elements = list(preds[0].get(PRED_FEATURE).elements)
    assert elements == ["GREETING"]
