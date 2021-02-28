from cassis import Cas

import nltk


from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE


class NltkStemmer(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        stemmer = nltk.PorterStemmer()

        # For every token, steam it and create an annotation in the CAS
        for cas_token in cas.select(TOKEN_TYPE):
            stem = stemmer.stem(cas_token.get_covered_text())
            begin = cas_token.begin
            end = begin + len(stem)
            prediction = create_prediction(cas, layer, feature, begin, end, stem)
            cas.add_annotation(prediction)
