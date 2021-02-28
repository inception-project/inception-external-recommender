import logging
from typing import List, Optional

import sklearn_crfsuite
from cassis import Cas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline


from ariadne.classifier import Classifier

from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE, SENTENCE_TYPE
from ariadne.protocol import TrainingDocument


logger = logging.getLogger(__name__)


class SklearnSentenceClassifier(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.debug("Start training for user [%s]", user_id)
        sentences = []
        targets = []

        for document in documents:
            cas = document.cas

            for sentence in cas.select(SENTENCE_TYPE):
                # Get the first annotation that covers the sentence
                annotations = cas.select_covered(layer, sentence)

                if len(annotations):
                    annotation = annotations[0]
                else:
                    continue

                assert (
                    sentence.begin == annotation.begin and sentence.end == annotation.end
                ), "Annotation should cover sentence fully!"

                label = getattr(annotation, feature)

                if label is None:
                    continue

                sentences.append(cas.get_covered_text(sentence))
                targets.append(label)

        model = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", MultinomialNB())])
        model.fit(sentences, targets)
        logger.debug(f"Training finished for user [%s]", user_id)

        self._save_model(user_id, model)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model: Optional[Pipeline] = self._load_model(user_id)

        if model is None:
            logger.debug("No trained model ready yet!")
            return

        for sentence in cas.select(SENTENCE_TYPE):
            predicted = model.predict([sentence.get_covered_text()])[0]
            prediction = create_prediction(cas, layer, feature, sentence.begin, sentence.end, predicted)
            cas.add_annotation(prediction)


# https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-use-conll-2002-data-to-build-a-ner-system
class SklearnMentionDetector(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.debug("Start training for user [%s]", user_id)
        sentences = []
        targets = []

        for document in documents:
            cas = document.cas

            for sentence in cas.select(SENTENCE_TYPE):
                tags = []
                words = []

                tokens = cas.select_covered(TOKEN_TYPE, sentence)
                annotations = list(cas.select_covered(layer, sentence))

                # Convert to BIO
                prev_tag = "O"
                for token in tokens:
                    is_inside = False
                    for annotation in annotations:
                        if token.begin >= annotation.begin and annotation.end:
                            is_inside = True
                            break

                    if not is_inside:
                        tag = "O"
                    elif prev_tag == "B-MENTION":
                        tag = "I-MENTION"
                    else:
                        tag = "B-MENTION"

                    prev_tag = tag

                    tags.append(tag)
                    words.append(token.get_covered_text())

                sentences.append(words)
                targets.append(tags)

        X_train = [self._sent2features(s) for s in sentences]
        y_train = targets

        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1)
        crf.fit(X_train, y_train)

        # model = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", MultinomialNB())])
        # model.fit(sentences, targets)
        logger.debug(f"Training finished for user [%s]", user_id)

        self._save_model(user_id, crf)

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model: Optional[sklearn_crfsuite.CRF] = self._load_model(user_id)

        if model is None:
            logger.debug("No trained model ready yet!")
            return

        all_tokens = []
        featurized_sentences = []

        for sentence in cas.select(SENTENCE_TYPE):
            tokens = list(cas.select_covered(TOKEN_TYPE, sentence))
            words = [token.get_covered_text() for token in tokens]

            all_tokens.append(tokens)
            featurized_sentences.append(self._sent2features(words))

        all_predictions = model.predict(featurized_sentences)

        assert len(all_predictions) == len(all_tokens)
        for predictions, tokens in zip(all_predictions, all_tokens):
            assert len(predictions) == len(tokens)

            begin = None
            end = None
            prev_tag = "O"
            for tag, token in zip(predictions, tokens):
                if begin is not None and end is not None:
                    if tag == "O" or (tag.startswith("B") and prev_tag.startswith("I")):
                        prediction = create_prediction(cas, layer, feature, begin, end, "X")
                        cas.add_annotation(prediction)

                if tag.startswith("B"):
                    begin = token.begin
                    end = token.end
                elif tag.startswith("I"):
                    end = token.end
                else:
                    begin = None
                    end = None

                prev_tag = tag

    def _sent2features(self, sent: List[str]):
        return [self._word2features(sent, i) for i in range(len(sent))]

    def _word2features(self, sent, i):
        word = sent[i]

        features = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word[-2:]": word[-2:],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
        }
        if i > 0:
            word1 = sent[i - 1]
            features.update(
                {
                    "-1:word.lower()": word1.lower(),
                    "-1:word.istitle()": word1.istitle(),
                    "-1:word.isupper()": word1.isupper(),
                }
            )
        else:
            features["BOS"] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1]
            features.update(
                {
                    "+1:word.lower()": word1.lower(),
                    "+1:word.istitle()": word1.istitle(),
                    "+1:word.isupper()": word1.isupper(),
                }
            )
        else:
            features["EOS"] = True

        return features
