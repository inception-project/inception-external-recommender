import html
from pathlib import Path
import zipfile

from sklearn.model_selection import train_test_split

from ariadne.util import setup_logging
from scripts.util import write_sentence_documents, download_file

PATH_ROOT: Path = Path(__file__).resolve().parents[1]
PATH_DATASETS = PATH_ROOT / "datasets"
PATH_DATASETS_SLS_ZIP = PATH_DATASETS / "sls.zip"
PATH_DATASETS_SLS = PATH_DATASETS / "sls"


def main():
    setup_logging()
    PATH_DATASETS_SLS.mkdir(exist_ok=True, parents=True)
    download_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment labelled sentences.zip",
        PATH_DATASETS_SLS_ZIP,
    )

    sentences = []
    labels = []
    with zipfile.ZipFile(PATH_DATASETS_SLS_ZIP) as myzip:
        with myzip.open("sentiment labelled sentences/sls_labelled.txt") as f:
            for i, line in enumerate(f):
                line = line.decode("utf-8")
                text, label = line.strip().split("\t")
                text = html.unescape(text).strip()
                label = "positive" if label == "1" else "negative"
                sentences.append(text)
                labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

    write_sentence_documents(X_train, y_train, PATH_DATASETS_SLS / f"sls_imdb_labeled.xmi")
    write_sentence_documents(X_test, y_test, PATH_DATASETS_SLS / f"sls_imdb_unlabeled.xmi", labeled=False)


if __name__ == "__main__":
    main()
