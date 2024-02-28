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
from flair.data import Sentence

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE, TOKEN_TYPE


class FlairNERClassifier(Classifier):
    def __init__(self, model_name: str, model_directory: Path = None, split_sentences: bool = True):
        super().__init__(model_directory=model_directory)
        self._model = Tagger.load(model_name)
        self._split_sentences = split_sentences

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str): 
        # Extract the sentences from the CAS
        if self._split_sentences:
            cas_sents = cas.select(SENTENCE_TYPE)
            sents = [Sentence(sent.get_covered_text(), use_tokenizer=False) for sent in cas_sents]
            offsets = [sent.begin for sent in cas_sents]

            # Find the named entities
            self._model.predict(sents)

            for offset, sent in zip(offsets, sents):
                # For every entity returned by spacy, create an annotation in the CAS
                for named_entity in sent.to_dict()["entities"]:
                    begin = named_entity["start_pos"] + offset
                    end = named_entity["end_pos"] + offset
                    label = named_entity["labels"][0]["value"]
                    prediction = create_prediction(cas, layer, feature, begin, end, label)
                    cas.add(prediction) 

        else:
            cas_tokens = cas.select(TOKEN_TYPE)

            # build sentence with correct whitespaces
            # (when using sentences, this should not be a problem afaik)
            text = ""
            last_end = 0
            for cas_token in cas_tokens:
                if cas_token.begin == last_end:
                    text += cas_token.get_covered_text()
                else:
                    text += " " + cas_token.get_covered_text()
                last_end = cas_token.end

            sent = Sentence(text, use_tokenizer=False)
            
            self._model.predict(sent)

            for named_entity in sent.to_dict()["entities"]:
                begin = named_entity["start_pos"]
                end = named_entity["end_pos"]
                label = named_entity["labels"][0]["value"]
                prediction = create_prediction(cas, layer, feature, begin, end, label)
                cas.add(prediction)
