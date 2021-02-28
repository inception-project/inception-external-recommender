from cassis import Cas

import jieba


from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction


class JiebaSegmenter(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        result = jieba.tokenize(cas.sofa_string)
        for tk in result:
            prediction = create_prediction(cas, layer, feature, tk[1], tk[2], tk[0])
            cas.add_annotation(prediction)
