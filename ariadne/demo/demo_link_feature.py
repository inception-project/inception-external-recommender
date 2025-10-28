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


class DemoLinkFeatureRecommender(Classifier):
    """Simple demo recommender that learns link roles between span annotations.

    Training
    --------
    For each document, we iterate over all annotations of the given
    ``layer`` and reads the ``feature`` field which is expected to contain
    link objects (UIMA link relations). It counts how often a source span
    text (lowercased) was linked to a particular target span text with a
    given role. The model stored per-user is a nested mapping:
    ``{source_text: {target_text: best_role}}``, where ``best_role`` is the
    role with the highest frequency for that (source, target) pair.

    Prediction
    ----------
    The ``predict`` method loads the per-user model and iterates over source
    annotations in the CAS. For each source whose lowercased covered text
    appears in the model, it looks for target annotations of the same
    ``layer`` inside the covering sentence. If a target's lowercased text
    matches a target recorded for the source, the recommender creates a span
    prediction suggestion that contains a link to the found target using the
    learned role. The suggestion is added to the CAS as a span prediction
    feature structure.
    """

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.info(
            f"Training triggered for [{feature}] on [{layer}] in [{len(documents)}] documents from project [{project_id}] for user [{user_id}]"
        )

        # Count how often each mention has been annotated with a given label
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for document in documents:
            cas = document.cas
            for annotation in cas.select(layer):
                source = annotation.get_covered_text().lower()
                links = annotation.get(feature)
                if links:
                    for link in links.elements:
                        target = link.target.get_covered_text().lower()
                        role = link.role.lower()
                        counts[source][target][role] += 1

        # Create a new dictionary that contains only the source/target/role with the highest count
        # for each link
        best_links = {
            source: {
                target: max(candidate_counts, key=candidate_counts.get) if candidate_counts else ""
                for target, candidate_counts in target_counts.items()
            }
            for source, target_counts in counts.items()
        }

        logger.info("Best labels: %s", best_links)
        self._save_model(user_id, best_links)

        logger.info("Training finished for user [%s]", user_id)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        logger.info(
            f"Prediction triggered on document [{document_id}] for [{feature}] on [{layer}] in project [{project_id}] for user [{user_id}]"
        )

        model = self._load_model(user_id)

        if model is None:
            return

        # Look for source annotations in the CAS and check if any of the mentions in the model correspond to the text starting
        # at that token
        for source in cas.select(layer):
            source_text = source.get_covered_text().lower()
            if source_text in model:
                sentence = list(
                    cas.select_covering("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", source)
                )
                if not sentence:
                    continue

                # Look for a suitable target annotation in the same sentence
                for target in cas.select_covered(layer, sentence[0]):
                    target_text = target.get_covered_text().lower()

                    # Source and target exist, create a link with the appropriate role
                    if target_text in model[source_text]:
                        role = model[source_text][target_text]
                        LinkType = cas.typesystem.get_type("custom.SpanLinksLink")
                        link = LinkType(role=role, target=target)
                        FSArray = cas.typesystem.get_type("uima.cas.FSArray")
                        links = FSArray(elements=[link])
                        suggestion = create_span_prediction(cas, layer, feature, source.begin, source.end, links)
                        cas.add(suggestion)

        logger.info("Prediction finished for user [%s]", user_id)
