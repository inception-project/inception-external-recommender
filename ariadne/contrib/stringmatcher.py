import logging
from itertools import chain
from typing import List

from cassis import Cas

from sklearn.preprocessing import LabelEncoder

from rust_fst import Map

import more_itertools as mit

from ariadne.classifier import Classifier
from ariadne.constants import TOKEN_TYPE
from ariadne.protocol import TrainingDocument

logger = logging.getLogger(__name__)


class LevenshteinStringMatcher(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.debug("Start training for user [%s]", user_id)

        mentions = []
        labels = []

        for document in documents:
            cas = document.cas
            for annotation in cas.select(layer):
                mention = annotation.get_covered_text()
                label = getattr(annotation, feature)

                mentions.append(mention)
                labels.append(label)

        le = LabelEncoder()
        le.fit(labels)

        items = [(k, v) for k, v in sorted(zip(mentions, le.transform(labels)))]

        # The map takes care of saving itself
        fst_path = self._get_fst_path(user_id)
        m = Map.from_iter(items, path=fst_path)

        logger.debug(f"Training finished for user [%s]", user_id)

        # We just save the LabelEncoder, the map saved itself
        self._save_model(user_id, le)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        le: LabelEncoder = self._load_model(user_id)

        if le is None:
            return

        fst_path = self._get_fst_path(user_id)
        m = Map(fst_path)

        # We iterate over the all candidates and check whether they match
        for (begin, end, term) in chain(self._generate_candidates(cas, 1), self._generate_candidates(cas, 2)):
            for mention, label_id in m.search(term=term, max_dist=2):
                label = le.inverse_transform([label_id])[0]
                prediction = self.create_prediction(cas, layer, feature, begin, end, label)
                cas.add_annotation(prediction)

    def _generate_candidates(self, cas: Cas, n: int):
        # We generate token n-grams
        for tokens in mit.windowed(cas.select(TOKEN_TYPE), n):
            begin = tokens[0].begin
            end = tokens[-1].end
            text = cas.sofa_string[begin:end]
            yield (begin, end, text)

    def _get_fst_path(self, user_id: str) -> str:
        p = self.model_directory / self.name / f"model_{user_id}.fst"
        return str(p)
