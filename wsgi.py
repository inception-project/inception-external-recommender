from ariadne.server import Server
from ariadne.util import setup_logging
from ariadne.contrib.spacy import SpacyNerClassifier

setup_logging()

server = Server()

server.add_classifier("spacy_ner", SpacyNerClassifier("en_core_web_sm"))
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

app = server._app

if __name__ == "__main__":
    server.start(debug=True, port=40022)
