from pathlib import Path
from typing import List

from cassis import TypeSystem, Cas

from ariadne.constants import SENTENCE_TYPE


def write_sentence_documents(sentences: List[str], labels: List[str], path: Path, labeled=True):
    typesystem = TypeSystem()
    cas = Cas(typesystem=typesystem)

    SentenceType = typesystem.create_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
    SentimentType = typesystem.create_type("webanno.custom.Sentiment")
    typesystem.add_feature(type_=SentimentType, name="value", rangeTypeName="uima.cas.String")

    cas.sofa_string = " ".join(sentences)

    begin = 0
    for sentence, label in zip(sentences, labels):
        end = begin + len(sentence)
        cas_sentence = SentenceType(begin=begin, end=end)
        sentiment_annotation = SentimentType(begin=begin, end=end, value=label)
        begin = end + 1

        cas.add_annotation(cas_sentence)

        if labeled:
            cas.add_annotation(sentiment_annotation)

    cas.to_xmi(path, pretty_print=True)

    for sentence in cas.select(SENTENCE_TYPE):
        print(cas.get_covered_text(sentence))
