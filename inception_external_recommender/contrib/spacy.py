from cassis import Cas

import spacy
from spacy.tokens import Doc


from inception_external_recommender.classifier import Classifier


class SpacyNerClassifier(Classifier):

    def __init__(self, model_name: str):
        self._model = spacy.load(model_name, disable=['parser'])

    def fit(self):
        pass

    def predict(self, document: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        tokens = self.get_tokens(document)
        words = [document.get_covered_text(token) for token in tokens]

        doc = Doc(self._model.vocab, words=words)

        # Find the named entities
        self._model.entity(doc)

        # For every entity returned by spacy, create an annotation in the CAS
        for ent in doc.ents:
            begin = tokens[ent.start].begin
            end = tokens[ent.end - 1].end
            label = ent.label_
            prediction = self.create_prediction(document, layer, feature, begin, end, label)
            document.add_annotation(prediction)


class SpacyPosClassifier(Classifier):

    def __init__(self, model_name: str):
        self._model = spacy.load(model_name, disable=['parser'])

    def fit(self):
        pass

    def predict(self, document: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        tokens = self.get_tokens(document)
        words = [document.get_covered_text(token) for token in tokens]

        doc = Doc(self._model.vocab, words=words)

        # Find the named entities
        self._model.tagger(doc)

        # For every token, extract the POS tag and create an annotation in the CAS
        for token in doc:
            begin = tokens[token.i].begin
            end = tokens[token.i].end
            label = token.pos_
            prediction = self.create_prediction(document, layer, feature, begin, end, label)
            document.add_annotation(prediction)
