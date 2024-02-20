from pathlib import Path

from cassis import Cas

from flair.nn import Classifier as Tagger
from flair.data import Sentence

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE


class FlairNERClassifier(Classifier):
    def __init__(self, model_name: str, model_directory: Path = None):
        super().__init__(model_directory=model_directory)
        self._model = Tagger.load(model_name)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the sentences from the CAS, we leave tokenization to flair
        cas_sents = cas.select(SENTENCE_TYPE)
        sents = [Sentence(sent.get_covered_text()) for sent in cas_sents]
        offsets = [sent.begin for sent in cas_sents]

        # Find the named entities
        self._model.predict(sents)

        for offset, sent in zip(offsets, sents):
            # For every entity returned by spacy, create an annotation in the CAS
            for named_entity in sent.to_dict()["entities"]:
                begin = named_entity["start_pos"] + offset
                end = named_entity["end_pos"] + offset
                label = named_entity["labels"][0]["value"]
                prediction = create_prediction(cas, layer, feature, begin, end, label)
                cas.add(prediction) 