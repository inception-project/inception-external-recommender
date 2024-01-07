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
from ariadne.contrib.inception_util import *

from cassis import Cas, TypeSystem


def test_create_prediction():
    typesystem = TypeSystem()
    Span = typesystem.create_type("custom.Span")
    typesystem.create_feature(Span, "inception_internal_predicted", "uima.cas.Boolean")
    typesystem.create_feature(Span, "value", "uima.cas.String")
    typesystem.create_feature(Span, "value_score", "uima.cas.Double")
    typesystem.create_feature(Span, "value_score_explanation", "uima.cas.String")
    typesystem.create_feature(Span, "value_auto_accept", "uima.cas.Boolean")
    cas = Cas(typesystem=typesystem)
    prediction = create_prediction(
        cas, "custom.Span", "value", 0, 4, "label", score=0.1, score_explanation="blah", auto_accept=True
    )
    assert prediction.get("begin") == 0
    assert prediction.get("end") == 4
    assert prediction.get("value") == "label"
    assert prediction.get("inception_internal_predicted") == True
    assert prediction.get("value_score") == 0.1
    assert prediction.get("value_score_explanation") == "blah"
    assert prediction.get("value_auto_accept") == True
