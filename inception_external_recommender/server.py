import json
from http import HTTPStatus
from typing import Dict

from flask import Flask, request, jsonify

from inception_external_recommender.classifier import Classifier
from inception_external_recommender.contrib.spacy import SpacyNerClassifier, SpacyPosClassifier
from inception_external_recommender.protocol import parse_prediction_request


class Server:

    def __init__(self):
        self._app = Flask(__name__)
        self._classifiers = {}

        self._app.add_url_rule("/<model_name>/predict", "predict", self._predict, methods=["POST"])
        self._app.add_url_rule("/<model_name>/train", "train", self._train, methods=["POST"])

        print(self._app.url_map)

    def add_classifier(self, name: str, classifier: Classifier):
        self._classifiers[name] = classifier

    def start(self):
        self._app.run(debug=True, host='0.0.0.0')

    def _predict(self, model_name: str):
        if model_name not in self._classifiers:
            return "Model with name [{0}] not found!".format(model_name), HTTPStatus.NOT_FOUND.value

        json_data = request.get_json()

        req = parse_prediction_request(json_data)
        classifier = self._classifiers[model_name]
        classifier.predict(req.document, req.layer, req.feature, req.project_id, req.document_id, req.user_id)

        result = jsonify(document=req.document.to_xmi())
        return result

    def _train(self, model_name: str):
        return model_name


if __name__ == '__main__':
    server = Server()
    server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
    server.add_classifier("spacy_pos", SpacyPosClassifier("en"))
    server.start()
