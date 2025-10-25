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

from simalign import SentenceAligner

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import SENTENCE_TYPE, create_relation_prediction

SPAN_ANNOTATION_TYPE = "webanno.custom.Base"


class SimAligner(Classifier):
    """
    Alignment of words in two sentences.

    The recommender assumes that there are exactly two sentences in the CAS.
    For each of the tokens, there must be an annotation of type `webanno.custom.Base`.
    The recommender then will predict relations between these base annotations.
    """

    def __init__(self):
        super().__init__()

        self._aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        sentences = cas.select(SENTENCE_TYPE)

        src_tokens = cas.select_covered(SPAN_ANNOTATION_TYPE, sentences[0])
        trg_tokens = cas.select_covered(SPAN_ANNOTATION_TYPE, sentences[1])

        src_sentence = [e.get_covered_text() for e in src_tokens]
        trg_sentence = [e.get_covered_text() for e in trg_tokens]

        alignments = self._aligner.get_word_aligns(src_sentence, trg_sentence)

        for matching_method in alignments:
            for source_idx, target_idx in alignments[matching_method]:
                prediction = create_relation_prediction(
                    cas, layer, feature, src_tokens[source_idx], trg_tokens[target_idx], ""
                )
                cas.add(prediction)
            break
