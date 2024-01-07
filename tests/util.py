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
from typing import List

from cassis import Cas, TypeSystem, merge_typesystems, load_typesystem, load_cas_from_xmi
from cassis.typesystem import TYPE_NAME_STRING, TYPE_NAME_BOOLEAN

from sklearn.datasets import fetch_20newsgroups

from ariadne.contrib.inception_util import SENTENCE_TYPE, IS_PREDICTION
from ariadne.protocol import TrainingDocument

PREDICTED_TYPE = "ariadne.testtype"
PREDICTED_FEATURE = "value"
USER = "test_user"
PROJECT_ID = "test_project"
NEWSGROUP_CATEGORIES = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]


def load_newsgroup_training_data() -> List[TrainingDocument]:
    twenty_train = fetch_20newsgroups(subset="train", categories=NEWSGROUP_CATEGORIES, shuffle=True, random_state=42)
    target_names = twenty_train.target_names

    typesystem = build_typesystem()
    Sentence = typesystem.get_type(SENTENCE_TYPE)
    PredictedType = typesystem.get_type(PREDICTED_TYPE)

    docs = []
    for i, (text, target) in enumerate(zip(twenty_train.data, twenty_train.target)):
        cas = Cas(typesystem=typesystem)
        cas.sofa_string = text

        begin = 0
        end = len(text)
        cas.add(Sentence(begin=begin, end=end))
        cas.add(PredictedType(begin=begin, end=end, value=target_names[target]))

        doc = TrainingDocument(cas, f"doc_{i}", USER)
        docs.append(doc)

    return docs


def load_newsgroup_test_data() -> List[Cas]:
    twenty_test = fetch_20newsgroups(subset="test", categories=NEWSGROUP_CATEGORIES, shuffle=True, random_state=42)

    typesystem = build_typesystem()
    Sentence = typesystem.get_type(SENTENCE_TYPE)

    result = []
    for text in twenty_test.data[:5]:
        cas = Cas(typesystem=typesystem)
        cas.sofa_string = text

        begin = 0
        end = len(text)
        cas.add(Sentence(begin=begin, end=end))

        result.append(cas)

    return result


def load_obama() -> Cas:
    # https://stackoverflow.com/a/20885799
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources

    from . import resources  # relative-import the *package* containing the templates

    with pkg_resources.open_binary(resources, "INCEpTION_TypeSystem.xml") as f:
        typesystem = merge_typesystems(load_typesystem(f), build_typesystem())

    with pkg_resources.open_binary(resources, "Wikipedia-Obama.xmi") as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)

    return cas


def build_typesystem() -> TypeSystem:
    typesystem = TypeSystem()
    typesystem.create_type(SENTENCE_TYPE)
    PredictedType = typesystem.create_type(PREDICTED_TYPE)
    typesystem.create_feature(PredictedType, PREDICTED_FEATURE, TYPE_NAME_STRING)
    typesystem.create_feature(PredictedType, IS_PREDICTION, TYPE_NAME_BOOLEAN)
    return typesystem
