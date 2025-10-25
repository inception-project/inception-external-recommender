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
from cassis import TypeSystem, Cas
from cassis.typesystem import TYPE_NAME_ANNOTATION, TYPE_NAME_STRING, TYPE_NAME_BOOLEAN, TYPE_NAME_DOUBLE
from ariadne.contrib.inception_util import SENTENCE_TYPE, IS_PREDICTION
from ariadne.demo.demo_relation import DemoRelationLayerRecommender


def test_demo_relation_recommender():
    typesystem = TypeSystem()

    # create sentence, a simple span annotation type and a relation type
    Sentence = typesystem.create_type(SENTENCE_TYPE)
    Span = typesystem.create_type("custom.Span")
    Relation = typesystem.create_type("custom.Relation")

    # features for relation: Governor and Dependent are annotations
    typesystem.create_feature(Relation, "Governor", TYPE_NAME_ANNOTATION)
    typesystem.create_feature(Relation, "Dependent", TYPE_NAME_ANNOTATION)
    typesystem.create_feature(Relation, "value", TYPE_NAME_STRING)
    typesystem.create_feature(Relation, IS_PREDICTION, TYPE_NAME_BOOLEAN)
    typesystem.create_feature(Relation, "value_score", TYPE_NAME_DOUBLE)

    # typesystem for CAS
    cas_train = Cas(typesystem)
    cas_train.sofa_string = "Alice and Bob"

    # create two span annotations and a sentence
    alice = Span(begin=0, end=5)
    bob = Span(begin=10, end=13)
    cas_train.add(alice)
    cas_train.add(bob)
    cas_train.add(Sentence(begin=0, end=len(cas_train.sofa_string)))

    # create two relation annotations with different labels between alice and bob
    r1 = Relation(begin=alice.begin, end=alice.end, Governor=alice, Dependent=bob, value="friend")
    r2 = Relation(begin=alice.begin, end=alice.end, Governor=alice, Dependent=bob, value="colleague")
    cas_train.add(r1)
    cas_train.add(r2)

    # wrap training doc list in the minimal structure expected by fit
    from ariadne.protocol import TrainingDocument

    docs = [TrainingDocument(cas_train, "doc1", "user1")]

    sut = DemoRelationLayerRecommender()
    sut.fit(docs, Relation.name, "value", None, "user1")

    # prediction CAS: same sentence and span annotations; no relation annotations yet
    cas_pred = Cas(typesystem)
    cas_pred.sofa_string = cas_train.sofa_string
    alice_p = Span(begin=0, end=5)
    bob_p = Span(begin=10, end=13)
    cas_pred.add(alice_p)
    cas_pred.add(bob_p)
    cas_pred.add(Sentence(begin=0, end=len(cas_pred.sofa_string)))

    sut.predict(cas_pred, Relation.name, "value", None, "doc1", "user1")

    # collect predicted relations
    preds = list(cas_pred.select(Relation.name))
    assert len(preds) == 2

    labels = sorted([p.get("value") for p in preds])
    assert labels == ["colleague", "friend"]

    # check scores: since we had one occurrence each, scores should be 0.5 each
    scores = sorted([p.get("value_score") for p in preds])
    assert scores == [0.5, 0.5]
