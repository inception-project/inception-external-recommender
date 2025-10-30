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

import logging

logger = logging.getLogger(__name__)


class DemoListTypesRecommender(Classifier):
    """Demo recommender that lists all annotation types present in the CAS

    This recommender does not produce span/label suggestions. Instead it
    inspects the CAS typesystem and counts how many annotations of each
    type are present. It logs the per-document counts during `fit` and
    logs the counts for the CAS passed to `predict`. For convenience the
    aggregate counts computed during `fit` are saved as a (simple) model.
    """

    def _count_types(self, cas: Cas):
        """Return a mapping type_name -> number of annotations of that type in the CAS.

        We are best-effort: if a type cannot be selected, it is skipped.
        """
        for t in cas.typesystem.get_types():
            items = list(cas.select(t.name))
            if len(items) > 0:
                logger.info("Counted type=%s count=%d", t.name, len(items))

    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.info(
            "Training triggered for listing types on layer=[%s] feature=[%s] in [%d] documents from project=[%s] for user=[%s]",
            layer,
            feature,
            len(documents),
            project_id,
            user_id,
        )

        for document in documents:
            cas = document.cas
            for dmd in cas.select("de.tudarmstadt.ukp.clarin.webanno.api.type.CASMetadata"):
                logger.info("Document: %s", dmd.sourceDocumentName)
            self._count_types(cas)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        logger.info(
            "Prediction triggered on document=[%s] for layer=[%s] feature=[%s] in project=[%s] for user=[%s]",
            document_id,
            layer,
            feature,
            project_id,
            user_id,
        )

        self._count_types(cas)
