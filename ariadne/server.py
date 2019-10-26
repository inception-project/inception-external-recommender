import json
import logging
from http import HTTPStatus
import threading

from filelock import Timeout
from flask import Flask, request, jsonify

from ariadne.classifier import Classifier
from ariadne.locking import LockManager
from ariadne.protocol import parse_prediction_request, parse_training_request

logger = logging.getLogger(__name__)


class Server:
    def __init__(self):
        self._app = Flask(__name__)
        self._classifiers = {}
        self._lock_manager = LockManager()

        self._app.add_url_rule("/<classifier_name>/predict", "predict", self._predict, methods=["POST"])
        self._app.add_url_rule("/<classifier_name>/train", "train", self._train, methods=["POST"])

        print(self._app.url_map)

    def add_classifier(self, name: str, classifier: Classifier):
        self._classifiers[name] = classifier

    def start(self, debug: bool = False):
        self._app.run(debug=debug, host="0.0.0.0")

    def _predict(self, classifier_name: str):
        logger.info("Got prediction request for [%s]", classifier_name)

        if classifier_name not in self._classifiers:
            return "Classifier with name [{0}] not found!".format(classifier_name), HTTPStatus.NOT_FOUND.value

        json_data = request.get_json()

        req = parse_prediction_request(json_data)
        classifier = self._classifiers[classifier_name]
        classifier.predict(req.cas, req.layer, req.feature, req.project_id, req.document_id, req.user_id)

        result = jsonify(document=req.cas.to_xmi())
        return result

    def _train(self, classifier_name: str):
        logger.info("Got training request for [%s]", classifier_name)

        if classifier_name not in self._classifiers:
            return "Classifier with name [{0}] not found!".format(classifier_name), HTTPStatus.NOT_FOUND.value

        json_data = request.get_json()
        req = parse_training_request(json_data)
        user_id = req.documents[0].user_id
        classifier = self._classifiers[classifier_name]

        try:
            with self._lock_manager.lock_model(classifier_name, user_id):
                # We spawn a thread and run the training in there so that this HTTP request can return directly
                args = (req.documents, req.layer, req.feature, req.project_id, user_id)
                threading.Thread(target=classifier.fit, args=args).start()
                return HTTPStatus.NO_CONTENT.description, HTTPStatus.NO_CONTENT.value
        except Timeout:
            logger.debug("Already training [%s] for user [%s], skipping!", classifier_name, user_id)
            return HTTPStatus.TOO_MANY_REQUESTS.description, HTTPStatus.TOO_MANY_REQUESTS.value