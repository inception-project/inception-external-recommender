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

import nltk

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE


class NltkStemmer(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        stemmer = nltk.PorterStemmer()

        # For every token, steam it and create an annotation in the CAS
        for cas_token in cas.select(TOKEN_TYPE):
            stem = stemmer.stem(cas_token.get_covered_text())
            begin = cas_token.begin
            end = begin + len(stem)
            prediction = create_prediction(cas, layer, feature, begin, end, stem)
            cas.add(prediction)
