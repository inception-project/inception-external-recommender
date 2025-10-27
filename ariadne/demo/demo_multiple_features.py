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
from typing import List, Dict

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument
from collections import defaultdict
from ariadne.contrib.inception_util import create_span_prediction, TOKEN_TYPE
from cassis.typesystem import TYPE_NAME_STRING

import logging

logger = logging.getLogger(__name__)


class DemoMultipleFeaturesRecommender(Classifier):
    """
    Demo recommender that behaves like DemoStringFeatureRecommender but ignores the
    provided `feature` parameter and instead trains/predicts on all string-typed
    features of the specified layer. A separate dictionary is learned for each
    feature (mapping mention -> best label).

    This recommender requires INCEpTION 39.0 or higher. Older versions of INCEpTION
    will only extract the configured feature, even though multiple features are trained
    and predicted.
    """

    def _get_string_features(self, cas: Cas, layer: str) -> List[str]:
        """Return the names of all features of `layer` whose range is a string."""
        try:
            AnnotationType = cas.typesystem.get_type(layer)
        except Exception:
            return []

        features = []
        for feat in AnnotationType.features:
            try:
                if feat.rangeType.name == TYPE_NAME_STRING:
                    features.append(feat.name)
            except Exception:
                # best-effort: skip features we cannot introspect
                continue

        return features

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.info(
            "Training triggered for all string features on [%s] in [%d] documents from project [%s] for user [%s]",
            layer,
            len(documents),
            project_id,
            user_id,
        )

        # counts: feature -> mention -> label -> count
        counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        features_discovered = None

        for document in documents:
            cas = document.cas

            if features_discovered is None:
                features_discovered = self._get_string_features(cas, layer)

            for annotation in cas.select(layer):
                mention = annotation.get_covered_text().lower()

                if not mention:
                    continue

                for feat in features_discovered or []:
                    label = annotation.get(feat)
                    if not label:
                        continue
                    counts[feat][mention][label] += 1

        # For each feature, compute best_labels mapping mention -> top label
        model: Dict[str, Dict[str, str]] = {}
        for feat, mention_map in counts.items():
            best_labels = {
                mention: max(candidate_counts, key=candidate_counts.get) if candidate_counts else ""
                for mention, candidate_counts in mention_map.items()
            }
            model[feat] = best_labels

        logger.info("Trained multiple-feature model for features: %s", list(model.keys()))
        self._save_model(user_id, model)

        logger.info("Training finished for user [%s]", user_id)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        logger.info(
            "Prediction triggered on document [%s] for all string features on [%s] in project [%s] for user [%s]",
            document_id,
            layer,
            project_id,
            user_id,
        )

        model = self._load_model(user_id)
        if model is None:
            return

        # Determine which string features to predict (use typesystem from cas)
        features = self._get_string_features(cas, layer)

        # For each token, try to predict for each discovered string feature if the token text
        # exists in the per-feature dictionary
        suggestion_count = 0
        for token in cas.select(TOKEN_TYPE):
            mention = token.get_covered_text().lower()
            for feat in features:
                feature_model = model.get(feat)
                if not feature_model:
                    continue
                if mention in feature_model:
                    label = feature_model.get(mention)
                    suggestion = create_span_prediction(
                        cas, layer, feat, token.begin, token.begin + len(mention), label
                    )
                    logger.info("Creating suggestion for feature [%s]: %s -> %s", feat, mention, label)
                    cas.add(suggestion)
                    suggestion_count += 1

        logger.info("Prediction finished for user [%s]; suggestions created: %d", user_id, suggestion_count)
