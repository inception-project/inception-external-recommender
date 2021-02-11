import logging
from pathlib import Path

import wget


def setup_logging():
    log_fmt = "%(process)d-%(thread)d %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    filelock_logger = logging.getLogger("filelock")
    filelock_logger.setLevel(logging.WARNING)
