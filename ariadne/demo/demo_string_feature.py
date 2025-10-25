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

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument
from collections import defaultdict
from ariadne.contrib.inception_util import create_span_prediction

import logging

logger = logging.getLogger(__name__)


class DemoStringFeatureRecommender(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.info(
            f"Training triggered for [{feature}] on [{layer}] in [{len(documents)}] documents from project [{project_id}] for user [{user_id}]"
        )

        # Count how often each mention has been annotated with a given label
        counts = defaultdict(lambda: defaultdict(int))

        for document in documents:
            cas = document.cas
            for annotation in cas.select(layer):
                mention = annotation.get_covered_text().lower()
                label = annotation.get(feature)

                if not mention or not label:
                    continue

                counts[mention][label] += 1

        # Create a new dictionary that contains only the label with the highest count for each mention
        best_labels = {
            mention: max(candidate_counts, key=candidate_counts.get) if candidate_counts else ""
            for mention, candidate_counts in counts.items()
        }

        logger.info(f"Best labels: {best_labels}")
        self._save_model(user_id, best_labels)

        logger.info("Training finished for user [%s]", user_id)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        logger.info(
            f"Prediction triggered on document [{document_id}] for [{feature}] on [{layer}] in project [{project_id}] for user [{user_id}]"
        )

        model = self._load_model(user_id)

        if model is None:
            return

        # For each token, check if any of the mentions in the model correspond to the text starting
        # at that token and create a new annotation if they do
        for token in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"):
            mention = token.get_covered_text().lower()
            if mention in model:
                label = model.get(mention)
                suggestion = create_span_prediction(cas, layer, feature, token.begin, token.begin + len(mention), label)
                cas.add(suggestion)

        logger.info("Prediction finished for user [%s]", user_id)
