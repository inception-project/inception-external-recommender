from cassis import Cas

import jieba


from ariadne.classifier import Classifier


class JiebaSegmenter(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        result = jieba.tokenize(cas.sofa_string)
        for tk in result:
            print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
            prediction = self.create_prediction(cas, layer, feature, tk[1], tk[2], tk[0])
            cas.add_annotation(prediction)

        # # For every token, extract the POS tag and create an annotation in the CAS
        # for cas_token, spacy_token in zip(self.iter_tokens(cas), doc):
        #     prediction = self.create_prediction(cas, layer, feature, cas_token.begin, cas_token.end, spacy_token.pos_)
        #     cas.add_annotation(prediction)
