from http import HTTPStatus

from flask import Flask, request, jsonify

from inception_external_recommender.classifier import Classifier
from inception_external_recommender.protocol import parse_prediction_request, parse_training_request


class Server:

    def __init__(self):
        self._app = Flask(__name__)
        self._classifiers = {}

        self._app.add_url_rule("/<classifier_name>/predict", "predict", self._predict, methods=["POST"])
        self._app.add_url_rule("/<classifier_name>/train", "train", self._train, methods=["POST"])

        print(self._app.url_map)

    def add_classifier(self, name: str, classifier: Classifier):
        self._classifiers[name] = classifier

    def start(self):
        self._app.run(debug=True, host='0.0.0.0')

    def _predict(self, classifier_name: str):
        if classifier_name not in self._classifiers:
            return "Classifier with name [{0}] not found!".format(classifier_name), HTTPStatus.NOT_FOUND.value

        json_data = request.get_json()

        req = parse_prediction_request(json_data)
        classifier = self._classifiers[classifier_name]
        classifier.predict(req.cas, req.layer, req.feature, req.project_id, req.document_id, req.user_id)

        result = jsonify(document=req.cas.to_xmi())
        return result

    def _train(self, classifier_name: str):
        if classifier_name not in self._classifiers:
            return "Classifier with name [{0}] not found!".format(classifier_name), HTTPStatus.NOT_FOUND.value

        json_data = request.get_json()
        req = parse_training_request(json_data)
        classifier = self._classifiers[classifier_name]
        classifier.fit(req.documents, req.layer, req.feature, req.project_id)

        return classifier_name
