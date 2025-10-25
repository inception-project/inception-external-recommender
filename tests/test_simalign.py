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
import re
from cassis import TypeSystem, Cas
from cassis.typesystem import TYPE_NAME_STRING, TYPE_NAME_ANNOTATION, TYPE_NAME_BOOLEAN
from ariadne.contrib.inception_util import SENTENCE_TYPE, IS_PREDICTION
from ariadne.contrib.simalign import SPAN_ANNOTATION_TYPE, SimAligner

RELATION_ANNOTATION_TYPE = "custom.Relation"


def test_predict():
    typesystem = TypeSystem()
    Sentence = typesystem.create_type(SENTENCE_TYPE)
    Span = typesystem.create_type(SPAN_ANNOTATION_TYPE)
    Relation = typesystem.create_type(RELATION_ANNOTATION_TYPE)
    typesystem.create_feature(Relation, "Governor", TYPE_NAME_ANNOTATION)
    typesystem.create_feature(Relation, "Dependent", TYPE_NAME_ANNOTATION)
    typesystem.create_feature(Relation, "value", TYPE_NAME_STRING)
    typesystem.create_feature(Relation, IS_PREDICTION, TYPE_NAME_BOOLEAN)

    cas = Cas(typesystem)
    cas.sofa_string = "I do like the color red. Red is the color that I like."
    for start, end in tokenize(cas.sofa_string):
        cas.add(Span(**{"begin": start, "end": end}))
    cas.add(Sentence(**{"begin": 0, "end": 24}))
    cas.add(Sentence(**{"begin": 25, "end": 54}))

    sut = SimAligner()
    sut.predict(cas, Relation.name, "value", None, None, None)

    pairs = [
        (r.get("Governor").get_covered_text(), r.get("Dependent").get_covered_text()) for r in cas.select(Relation)
    ]
    assert set(pairs) == set(
        [
            ("red", "Red"),
            ("do", "is"),
            ("the", "the"),
            ("color", "color"),
            ("I", "I"),
            ("like", "like"),
            (".", "."),
        ]
    )


def tokenize(string):
    positions = []
    for match in re.compile(r"\w+|\S").finditer(string):
        positions.append((match.start(), match.end()))
    return positions
