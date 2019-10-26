import logging
from pathlib import Path

import wget

from ariadne import model_directory


def setup_logging():
    log_fmt = "%(process)d-%(thread)d %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    filelock_logger = logging.getLogger("filelock")
    filelock_logger.setLevel(logging.WARNING)


def download_file(url: str, target_path: Path):
    import ssl

    if target_path.exists():
        logging.info("File already exists: [%s]", str(target_path.resolve()))
        return

    wget.download(url, str(target_path.resolve()))



