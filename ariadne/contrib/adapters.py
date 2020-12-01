from collections import Counter
from pathlib import Path
from typing import List, Dict

from cassis import Cas
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification

from ariadne.classifier import Classifier
from ariadne.constants import SENTENCE_TYPE, TOKEN_TYPE

import torch

import numpy as np


class AdapterSequenceTagger(Classifier):
    def __init__(self, base_model_name: str, adapter_name: str, labels: List[str], model_directory: Path = None):
        super().__init__(model_directory=model_directory)
        self._labels = labels
        self._label_map = {i: label for i, label in enumerate(labels)}

        self._model_name = base_model_name
        self._adapter_name = adapter_name
        self._adapter_internal_name = None

        self._model = self._build_model()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        for sentence in cas.select(SENTENCE_TYPE):
            cas_tokens = list(cas.select_covered(TOKEN_TYPE, sentence))
            tokens = [t.get_covered_text() for t in cas_tokens]

            grouped_bert_tokens = self._tokenize_bert(tokens)
            predictions = self._predict(grouped_bert_tokens)

            grouped_predictions = self._align_tokens(tokens, grouped_bert_tokens, predictions)

            for token, grouped_prediction in zip(cas_tokens, grouped_predictions):
                begin = token.begin
                end = token.end
                label = Counter([self._label_map[pred] for pred in grouped_prediction]).most_common(1)[0][0]
                prediction = self.create_prediction(cas, layer, feature, begin, end, label)
                cas.add_annotation(prediction)

    def _tokenize_bert(self, cas_tokens: List[str]) -> List[torch.Tensor]:
        grouped_bert_tokens = [torch.LongTensor([self._tokenizer.cls_token_id])]

        for cas_token in cas_tokens:
            tokens = self._tokenizer.encode(
                cas_token, return_tensors="pt", add_special_tokens=False, max_length=self._tokenizer.max_len
            )
            grouped_bert_tokens.append(tokens.squeeze(axis=0))

        grouped_bert_tokens.append(torch.LongTensor([self._tokenizer.sep_token_id]))

        return grouped_bert_tokens

    def _predict(self, grouped_bert_tokens: List[torch.Tensor]) -> torch.Tensor:
        flattened_bert_tokens = torch.cat(grouped_bert_tokens)
        flattened_bert_tokens = torch.unsqueeze(flattened_bert_tokens, 0)

        preds = self._model(flattened_bert_tokens, adapter_names=[self._adapter_name])[0]
        preds = preds.detach().numpy()
        preds = np.argmax(preds, axis=2)

        return preds.squeeze()

    def _align_tokens(
        self, cas_tokens: List[str], grouped_bert_tokens: List[torch.Tensor], predictions: torch.Tensor
    ) -> List[torch.Tensor]:
        grouped_bert_tokens = grouped_bert_tokens[1:-1]
        predictions = predictions[1:-1]

        sizes = [len(group) for group in grouped_bert_tokens]
        grouped_predictions = []

        assert len(predictions) == sum(sizes)

        ptr = 0
        for size in sizes:
            group = predictions[ptr : ptr + size]
            assert len(group) == size

            grouped_predictions.append(group)
            ptr += size

        assert len(cas_tokens) == len(grouped_predictions)

        return grouped_predictions

    def _build_model(self):
        config = AutoConfig.from_pretrained(
            self._model_name,
            num_labels=len(self._labels),
            id2label=self._label_map,
            label2id={label: i for i, label in enumerate(self._labels)},
        )
        model = AutoModelForTokenClassification.from_pretrained(self._model_name, config=config)
        self._adapter_internal_name = model.load_adapter(self._adapter_name, "text_task")
        return model


class AdapterSentenceClassifier(Classifier):
    def __init__(self, base_model_name: str, adapter_name: str, labels: List[str], model_directory: Path = None):
        super().__init__(model_directory=model_directory)
        self._labels = labels
        self._label_map = {i: label for i, label in enumerate(labels)}

        self._model_name = base_model_name
        self._adapter_name = adapter_name
        self._adapter_internal_name = None

        self._model = self._build_model()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        for i, sentence in enumerate(cas.select(SENTENCE_TYPE)):
            token_ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(sentence.get_covered_text()))
            input_tensor = torch.tensor([token_ids])

            # predict output tensor
            outputs = self._model(input_tensor, adapter_names=[self._adapter_internal_name])

            # retrieve the predicted class label
            label_id = torch.argmax(outputs[0]).item()
            label = self._label_map[label_id]
            prediction = self.create_prediction(cas, layer, feature, sentence.begin, sentence.end, label)
            cas.add_annotation(prediction)

    def _build_model(self):
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self._adapter_internal_name = model.load_adapter(self._adapter_name, "text_task")
        return model
