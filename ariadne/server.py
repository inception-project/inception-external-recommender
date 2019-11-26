import logging
import tempfile
from http import HTTPStatus
import threading
from pathlib import Path
from typing import Dict

from filelock import Timeout, FileLock
from flask import Flask, request, jsonify

from ariadne.classifier import Classifier
from ariadne.protocol import parse_prediction_request, parse_training_request

logger = logging.getLogger(__name__)


class Server:
    def __init__(self):
        self._app = Flask(__name__)
        self._classifiers: Dict[str, Classifier] = {}
        self._lock_directory: Path = Path(tempfile.gettempdir()) / ".ariadne_locks"

        self._app.add_url_rule("/<classifier_name>/predict", "predict", self._predict, methods=["POST"])
        self._app.add_url_rule("/<classifier_name>/train", "train", self._train, methods=["POST"])

    def add_classifier(self, name: str, classifier: Classifier):
        self._classifiers[name] = classifier

    def start(self, debug: bool = False, host: str = "0.0.0.0", port: int = 5000):
        self._app.run(debug=debug, host=host, port=port)

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
        user_id = req.user_id
        classifier = self._classifiers[classifier_name]

        try:
            # The lock needs to be acquired out here, not in the fn scope, else it would
            # just throw the Timeout inside fn.
            lock = self._get_lock(classifier.name, user_id)
            lock.acquire()

            def _fn():
                try:
                    classifier.fit(req.documents, req.layer, req.feature, req.project_id, user_id)
                finally:
                    lock.release()

            # We spawn a thread and run the training in there so that this HTTP request can return directly
            threading.Thread(target=_fn, daemon=True).start()
            return HTTPStatus.NO_CONTENT.description, HTTPStatus.NO_CONTENT.value
        except Timeout:
            logger.info("Already training [%s] for user [%s], skipping!", classifier_name, user_id)
            return HTTPStatus.TOO_MANY_REQUESTS.description, HTTPStatus.TOO_MANY_REQUESTS.value

    def _get_lock(self, classifier_name: str, user_id: str) -> FileLock:
        self._lock_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_directory / f"{classifier_name}_{user_id}.lock"
        return FileLock(lock_path, timeout=1)
