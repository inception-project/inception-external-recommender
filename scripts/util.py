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
import logging
from pathlib import Path
from typing import List

import wget
from cassis import TypeSystem, Cas

from ariadne.constants import SENTENCE_TYPE


def download_file(url: str, target_path: Path):
    import ssl

    if target_path.exists():
        logging.info("File already exists: [%s]", str(target_path.resolve()))
        return

    wget.download(url, str(target_path.resolve()))


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
