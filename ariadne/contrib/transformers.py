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

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction
from cassis import Cas


class TransformerNerClassifier(Classifier):
    def __init__(self, model_name: str):
        super().__init__()
        # Load the Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="first")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        document_text = cas.sofa_string
        predictions = self.ner_pipeline(document_text)
        for prediction in predictions:
            start_char = prediction["start"]
            end_char = prediction["end"]
            label = prediction["entity_group"]
            cas_prediction = create_prediction(cas, layer, feature, start_char, end_char, label)
            cas.add(cas_prediction)
