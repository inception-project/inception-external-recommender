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

import spacy
from spacy.tokens import Doc


from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE


class SpacyNerClassifier(Classifier):
    def __init__(self, model_name: str, model_directory: Path = None):
        super().__init__(model_directory=model_directory)
        try:
            self._model = spacy.load(model_name, disable=["parser"])
        except OSError:
            print(f"Downloading {model_name}...")
            spacy.cli.download(model_name)
            self._model = spacy.load(model_name, disable=["parser"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        cas_tokens = cas.select(TOKEN_TYPE)
        words = [cas.get_covered_text(cas_token) for cas_token in cas_tokens]

        doc = Doc(self._model.vocab, words=words)

        # Find the named entities
        self._model.get_pipe("ner")(doc)

        # For every entity returned by spacy, create an annotation in the CAS
        for named_entity in doc.ents:
            begin = cas_tokens[named_entity.start].begin
            end = cas_tokens[named_entity.end - 1].end
            label = named_entity.label_
            prediction = create_prediction(cas, layer, feature, begin, end, label)
            cas.add_annotation(prediction)


class SpacyPosClassifier(Classifier):
    def __init__(self, model_name: str):
        super().__init__()
        try:
            self._model = spacy.load(model_name, disable=["parser"])
        except OSError:
            print(f"Downloading {model_name}...")
            spacy.cli.download(model_name)
            self._model = spacy.load(model_name, disable=["parser"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        words = [cas.get_covered_text(cas_token) for cas_token in cas.select(TOKEN_TYPE)]

        doc = Doc(self._model.vocab, words=words)

        # Get the pos tags
        self._model.get_pipe("tok2vec")(doc)
        self._model.get_pipe("tagger")(doc)

        # For every token, extract the POS tag and create an annotation in the CAS
        for cas_token, spacy_token in zip(cas.select(TOKEN_TYPE), doc):
            prediction = create_prediction(cas, layer, feature, cas_token.begin, cas_token.end, spacy_token.tag_)
            cas.add_annotation(prediction)
