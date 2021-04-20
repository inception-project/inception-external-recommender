from cassis import Cas

import jieba
from simalign import SentenceAligner

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE, TOKEN_TYPE


class SimAligner(Classifier):
    def __init__(self):
        super().__init__()

        self._aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):

        sentences = cas.select(SENTENCE_TYPE)

        src_tokens = cas.select_covered("webanno.custom.Base", sentences[0])
        trg_tokens = cas.select_covered("webanno.custom.Base", sentences[1])

        src_sentence = [e.get_covered_text() for e in src_tokens]
        trg_sentence = [e.get_covered_text() for e in trg_tokens]

        print(src_sentence)
        print(trg_sentence)

        alignments = self._aligner.get_word_aligns(src_sentence, trg_sentence)

        Relation = cas.typesystem.get_type(layer)
        print(list(Relation.all_features))

        for matching_method in alignments:
            for source_idx, target_idx in alignments[matching_method]:
                src = src_tokens[source_idx]
                target = trg_tokens[target_idx]
                prediction = Relation(
                    Governor=src,
                    Dependent=target,
                    begin=target.begin,
                    end=target.end,
                    inception_internal_predicted=True,
                )
                # setattr(prediction, feature, f"{src.get_covered_text()} -> {target.get_covered_text()}")
                setattr(prediction, feature, "")
                print(source_idx, target_idx, prediction)

                cas.add_annotation(prediction)
            break
