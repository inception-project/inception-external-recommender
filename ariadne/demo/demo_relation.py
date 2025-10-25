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
from typing import List, Tuple, Dict

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument
from collections import defaultdict, Counter
from ariadne.contrib.inception_util import create_relation_prediction, SENTENCE_TYPE

import logging

logger = logging.getLogger(__name__)

FEAT_REL_SOURCE = "Governor"
FEAT_REL_TARGET = "Dependent"


class DemoRelationLayerRecommender(Classifier):
    """
    Simple demo recommender for relation layers.

    Training builds a map from (source_text, target_text) -> list of observed relation labels.
    Prediction looks for pairs of annotations that co-occur inside the same sentence and emits
    relation predictions with a score equal to label frequency / total occurrences for that pair.
    """

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.info(
            "Training triggered for [%s] on [%s] in [%d] documents from project [%s] for user [%s]",
            feature,
            layer,
            len(documents),
            project_id,
            user_id,
        )

        # model: Dict[Tuple[source_text, target_text], List[label]]
        model: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for document in documents:
            cas = document.cas
            # iterate over existing relation annotations of the predicted layer
            for rel in cas.select(layer):
                source = rel.get(FEAT_REL_SOURCE)
                target = rel.get(FEAT_REL_TARGET)
                label = rel.get(feature)

                if source is None or target is None or not label:
                    continue

                source_text = source.get_covered_text().lower().strip()
                target_text = target.get_covered_text().lower().strip()

                model[(source_text, target_text)].append(label)

        logger.info("Trained relation model entries: %d", len(model))
        self._save_model(user_id, model)

        logger.info("Training finished for user [%s]", user_id)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        logger.info(
            "Prediction triggered on document [%s] for [%s] on [%s] in project [%s] for user [%s]",
            document_id,
            feature,
            layer,
            project_id,
            user_id,
        )

        # model is a mapping from (source_text, target_text) -> [labels]
        model = self._load_model(user_id)
        if model is None:
            return

        # Determine attach type which indicates the endpoint candiates for our relations
        predicted_type = cas.typesystem.get_type(layer)
        attach_type_name = predicted_type.get_feature(FEAT_REL_SOURCE).rangeType.name

        # Iterate over sentence sample units (predict relations within sentences)
        for unit in cas.select(SENTENCE_TYPE):
            candidates = list(cas.select_covered(attach_type_name, unit))

            # for each ordered pair of distinct annotations, check model and emit predictions
            for i, source in enumerate(candidates):
                for j, target in enumerate(candidates):
                    if i == j:
                        continue

                    source_text = source.get_covered_text().lower().strip()
                    target_text = target.get_covered_text().lower().strip()

                    key = (source_text, target_text)
                    occurrences = model.get(key)
                    if not occurrences:
                        continue

                    counts = Counter(occurrences)
                    total = sum(counts.values())

                    for relation_label, cnt in counts.items():
                        score = cnt / total if total > 0 else 0.0
                        prediction = create_relation_prediction(
                            cas, layer, feature, source, target, relation_label, score
                        )
                        cas.add(prediction)

        logger.info("Prediction finished for user [%s]", user_id)
