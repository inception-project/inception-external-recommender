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

import jieba

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction


class JiebaSegmenter(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        result = jieba.tokenize(cas.sofa_string)
        for tk in result:
            prediction = create_prediction(cas, layer, feature, tk[1], tk[2], tk[0])
            cas.add(prediction)
