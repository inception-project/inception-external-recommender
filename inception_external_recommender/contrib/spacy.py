from cassis import Cas

import spacy
from spacy.tokens import Doc


from inception_external_recommender.classifier import Classifier


class SpacyNerClassifier(Classifier):

    def __init__(self, model_name: str):
        self._model = spacy.load(model_name, disable=['parser'])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        cas_tokens = self.get_tokens(cas)
        words = [cas.get_covered_text(cas_token) for cas_token in cas_tokens]

        doc = Doc(self._model.vocab, words=words)

        # Find the named entities
        self._model.entity(doc)

        # For every entity returned by spacy, create an annotation in the CAS
        for named_entity in doc.ents:
            begin = cas_tokens[named_entity.start].begin
            end = cas_tokens[named_entity.end - 1].end
            label = named_entity.label_
            prediction = self.create_prediction(cas, layer, feature, begin, end, label)
            cas.add_annotation(prediction)


class SpacyPosClassifier(Classifier):

    def __init__(self, model_name: str):
        self._model = spacy.load(model_name, disable=['parser'])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        words = [cas.get_covered_text(cas_token) for cas_token in self.iter_tokens(cas)]

        doc = Doc(self._model.vocab, words=words)

        # Find the named entities
        self._model.tagger(doc)

        # For every token, extract the POS tag and create an annotation in the CAS
        for cas_token, spacy_token in zip(self.iter_tokens(cas), doc):
            prediction = self.create_prediction(cas, layer, feature, cas_token.begin, cas_token.end, spacy_token.pos_)
            cas.add_annotation(prediction)
