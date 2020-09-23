from cassis import Cas

import jieba


from ariadne.classifier import Classifier


class JiebaSegmenter(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        result = jieba.tokenize(cas.sofa_string)
        for tk in result:
            prediction = self.create_prediction(cas, layer, feature, tk[1], tk[2], tk[0])
            cas.add_annotation(prediction)
