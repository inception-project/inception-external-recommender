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
import logging
from collections import defaultdict
from itertools import chain
from typing import List

from cassis import Cas

from sklearn.preprocessing import LabelEncoder

from rust_fst import Map

import more_itertools as mit

from ariadne.classifier import Classifier

from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE
from ariadne.protocol import TrainingDocument

logger = logging.getLogger(__name__)


class LevenshteinStringMatcher(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.debug("Start training for user [%s]", user_id)

        mentions = []
        labels = []

        counts = defaultdict(lambda: defaultdict(int))

        for document in documents:
            cas = document.cas
            for annotation in cas.select(layer):
                mention = annotation.get_covered_text().lower()
                label = getattr(annotation, feature)

                if not label:
                    continue

                counts[mention][label] += 1

        # Just use the entity that was most often linked with this mention
        for mention, candidates in counts.items():
            if candidates:
                label = max(candidates, key=candidates.get)
            else:
                label = ""

            mentions.append(mention)
            labels.append(label)

        le = LabelEncoder()
        le.fit(labels)

        items = [(k, v) for k, v in sorted(zip(mentions, le.transform(labels)))]

        logger.debug("Training finished for user [%s]", user_id)

        self._save_model(user_id, (le, items))

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model = self._load_model(user_id)

        if model is None:
            return

        le, items = model

        m = Map.from_iter(items)

        # We iterate over the all candidates and check whether they match
        for begin, end, term in chain(
            self._generate_candidates(cas, 3), self._generate_candidates(cas, 2), self._generate_candidates(cas, 1)
        ):
            for mention, label_id in m.search(term=term, max_dist=2):
                label = le.inverse_transform([label_id])[0]
                prediction = create_prediction(cas, layer, feature, begin, end, label)
                cas.add(prediction)

    def _generate_candidates(self, cas: Cas, n: int):
        # We generate token n-grams
        for tokens in mit.windowed(cas.select(TOKEN_TYPE), n):
            begin = tokens[0].begin
            end = tokens[-1].end
            text = cas.sofa_string[begin:end]
            yield begin, end, text
