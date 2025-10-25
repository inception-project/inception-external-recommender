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
import random
from pathlib import Path
import tarfile

from ariadne.util import setup_logging
from scripts.util import write_sentence_documents, download_file

PATH_ROOT: Path = Path(__file__).resolve().parents[1]
PATH_DATASETS = PATH_ROOT / "datasets"
PATH_DATASETS_IMDB = PATH_DATASETS / "imdb.tar.gz"
PATH_DATASETS_IMDB_EXTRACTED = PATH_DATASETS / "imdb"
PATH_DATASETS_IMDB_TRAIN = PATH_DATASETS_IMDB_EXTRACTED / "aclImdb" / "train"


def read_data(documents):
    sentences = []
    labels = []

    for p, label in documents:
        with p.open() as f:
            text = f.read()
            sentences.append(text)
            labels.append(label)

    return sentences, labels


def main():
    setup_logging()
    PATH_DATASETS.mkdir(exist_ok=True, parents=True)
    download_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", PATH_DATASETS_IMDB)

    if not PATH_DATASETS_IMDB_EXTRACTED.exists():
        with tarfile.open(PATH_DATASETS_IMDB) as mytar:
            import os

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(mytar, PATH_DATASETS_IMDB_EXTRACTED)

    positive = [(p, "positive") for p in (PATH_DATASETS_IMDB_TRAIN / "pos").iterdir()]
    negative = [(p, "negative") for p in (PATH_DATASETS_IMDB_TRAIN / "neg").iterdir()]
    unsup = [(p, "unsup") for p in (PATH_DATASETS_IMDB_TRAIN / "unsup").iterdir()]
    unsup = random.sample(unsup, 100)

    docs = random.sample(positive, 200) + random.sample(negative, 200)

    for i in range(10):
        random.shuffle(docs)

    sentences_per_doc = 200
    for idx, i in enumerate(range(0, len(docs), sentences_per_doc)):
        slice = docs[i : i + sentences_per_doc]
        sentences, labels = read_data(slice)

        doc_name = PATH_DATASETS_IMDB_EXTRACTED / f"imdb_{idx}_labeled.xmi"
        write_sentence_documents(sentences, labels, doc_name)

    sentences, labels = read_data(unsup)
    write_sentence_documents(sentences, labels, PATH_DATASETS_IMDB_EXTRACTED / "imdb_unlabeled.xmi", labeled=False)


if __name__ == "__main__":
    main()
