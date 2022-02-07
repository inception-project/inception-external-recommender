"""
@author: Ghadeer Mobasher
"""
from cassis import Cas
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from ariadne.classifier import Classifier

class HuggingFaceClassifier(Classifier):
    '''As an example, to use it HuggingFace models for token classification  
    HuggingFaceClassifier(model_name="ghadeermobasher/BC5CDR-Chemical-Disease-balanced-pubmedbert")'''
    def __init__(self, model_name: str):
            super().__init__()
            self._model = model_name
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        tokenizer = AutoTokenizer.from_pretrained(self._model)
        model = AutoModelForTokenClassification.from_pretrained(self._model)
        nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer,aggregation_strategy="max")
        for sentence in  cas.select(SENTENCE_TYPE):
            columns = {'word', 'start', 'end', 'entity_group', 'score'}
            df = pd.DataFrame(columns=columns)
            s=nlp_ner(sentence.get_covered_text())
            for item in s:
                 df = df.append(item, ignore_index= True)
            for i in range(len(df)):
               prediction = create_prediction(cas, layer, feature,df.loc[i, "start"]+sentence.begin,df.loc[i, "end"]+sentence.begin,df.loc[i, "entity_group"])
               cas.add_annotation(prediction)
