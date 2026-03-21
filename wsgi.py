# Licensed to the Technische Universität Darmstadt under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The Technische Universität Darmstadt
# licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lazily create the server and register classifiers at first use.
# This avoids importing heavy native/Objective-C-backed libraries
# (spacy, transformers, etc.) in the master process before gunicorn forks
# worker processes — which can trigger macOS objc initialize-after-fork errors.
import logging
import os
import threading

logger = logging.getLogger(__name__)


def create_server():
    logger.debug("create_server() running in pid=%s threads=%s", os.getpid(), [t.name for t in threading.enumerate()])
    from ariadne.demo.demo_link_feature import DemoLinkFeatureRecommender
    from ariadne.demo.demo_multiple_features import DemoMultipleFeaturesRecommender
    from ariadne.demo.demo_relation import DemoRelationLayerRecommender
    from ariadne.demo.demo_string_array_feature import DemoStringArrayFeatureRecommender
    from ariadne.demo.demo_string_feature import DemoStringFeatureRecommender
    from ariadne.demo.demo_list_types import DemoListTypesRecommender
    from ariadne.server import Server
    from ariadne.util import setup_logging

    setup_logging()

    server = Server()

    server.add_classifier("demo_string_feature", DemoStringFeatureRecommender())
    server.add_classifier("demo_string_array_feature", DemoStringArrayFeatureRecommender())
    server.add_classifier("demo_link_feature", DemoLinkFeatureRecommender())
    server.add_classifier("demo_relation_layer", DemoRelationLayerRecommender())
    server.add_classifier("demo_multiple_features", DemoMultipleFeaturesRecommender())
    server.add_classifier("demo_list_types", DemoListTypesRecommender())

    # Example registration for GLiNER2 span recommender. Replace the model
    # identifier if you want a different pretrained weight. The default model
    # used by the adapter is `fastino/gliner2-base-v1` when no instance is passed.
    # server.add_classifier("gliner2_span", Gliner2SpanRecommender("fastino/gliner2-base-v1"))

    # server.add_classifier("spacy_pos", SpacyPosClassifier("en_core_web_sm"))
    # server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
    # server.add_classifier("jieba", JiebaSegmenter())
    # server.add_classifier("stemmer", NltkStemmer())
    # server.add_classifier("leven", LevenshteinStringMatcher())
    # server.add_classifier("sbert", SbertSentenceClassifier())
    # server.add_classifier(
    #     "adapter_pos",
    #     AdapterSequenceTagger(
    #         base_model_name="bert-base-uncased",
    #         adapter_name="pos/ldc2012t13@vblagoje",
    #         labels=[
    #             "ADJ",
    #             "ADP",
    #             "ADV",
    #             "AUX",
    #             "CCONJ",
    #             "DET",
    #             "INTJ",
    #             "NOUN",
    #             "NUM",
    #             "PART",
    #             "PRON",
    #             "PROPN",
    #             "PUNCT",
    #             "SCONJ",
    #             "SYM",
    #             "VERB",
    #             "X",
    #         ],
    #     ),
    # )
    #
    # server.add_classifier(
    #     "adapter_sent",
    #     AdapterSentenceClassifier(
    #         "bert-base-multilingual-uncased",
    #         "sentiment/hinglish-twitter-sentiment@nirantk",
    #         labels=["negative", "positive"],
    #         config="pfeiffer",
    #     ),
    # )
    return server


# `_LazyApp` is a lightweight WSGI application proxy that defers creating the
# full `Server` and importing heavy NLP libraries until the first incoming
# request. This keeps module import cheap (avoid initializing spaCy/transformers
# or native/Objective-C-backed subsystems in the master process) and prevents
# initialize-after-fork issues on platforms like macOS when using gunicorn.
#
# Note: initialization may be triggered by concurrent requests — make `_init`
# thread-safe if you add concurrent worker startup paths (e.g., use a lock).
class _LazyApp:
    def __init__(self):
        self._app = None
        self._server = None
        # Protect initialization from concurrent calls within a single
        # worker process (e.g., threaded workers). Double-checked locking
        # ensures only one thread performs the heavy server creation.
        self._init_lock = threading.Lock()

    # Module-import diagnostic (non-blocking, uses module-level logger)
    logger.debug("wsgi imported in pid=%s threads=%s", os.getpid(), [t.name for t in threading.enumerate()])

    def _init(self):
        if self._app is None:
            with self._init_lock:
                if self._app is None:
                    self._server = create_server()
                    self._app = self._server._app

    def __call__(self, environ, start_response):
        self._init()
        return self._app(environ, start_response)


app = _LazyApp()

if __name__ == "__main__":
    srv = create_server()
    srv.start(debug=True, port=40022)
