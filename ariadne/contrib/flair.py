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
from pathlib import Path

from cassis import Cas

from flair.nn import Classifier as Tagger
from flair.data import Sentence, Token

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE, TOKEN_TYPE


def fix_whitespaces(cas_tokens):
    tokens = []
    for cas_token, following_cas_token in zip(cas_tokens, cas_tokens[1:] + [None]):
        if following_cas_token is not None:
            dist = following_cas_token.begin - cas_token.end
        else:
            dist = 1
        token = Token(cas_token.get_covered_text(), whitespace_after=dist, start_position=cas_token.begin)
        tokens.append(token)
    return tokens


class FlairNERClassifier(Classifier):
    def __init__(self, model_name: str, model_directory: Path = None, split_sentences: bool = True):
        super().__init__(model_directory=model_directory)
        self._model = Tagger.load(model_name)
        self._split_sentences = split_sentences

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the sentences from the CAS
        if self._split_sentences:
            sentences = []
            cas_sents = cas.select(SENTENCE_TYPE)
            for cas_sent in cas_sents:
                # transform cas tokens to flair tokens with correct spacing
                cas_tokens = cas.select_covered(TOKEN_TYPE, cas_sent)
                tokens = fix_whitespaces(cas_tokens)
                sentences.append(Sentence(tokens))

            # Find the named entities
            self._model.predict(sentences)

            for sentence in sentences:
                # For every entity returned by spacy, create an annotation in the CAS
                for named_entity in sentence.get_spans():
                    begin = named_entity.start_position
                    end = named_entity.end_position
                    label = named_entity.tag
                    prediction = create_prediction(cas, layer, feature, begin, end, label)
                    cas.add(prediction)

        else:
            cas_tokens = cas.select(TOKEN_TYPE)
            text = fix_whitespaces(cas_tokens)
            sent = Sentence(text)

            self._model.predict(sent)

            for named_entity in sent.get_spans():
                begin = named_entity.start_position
                end = named_entity.end_position
                label = named_entity.tag
                prediction = create_prediction(cas, layer, feature, begin, end, label)
                cas.add(prediction)
