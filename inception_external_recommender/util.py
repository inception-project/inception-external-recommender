import logging
from pathlib import Path

import wget


def setup_logging():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)


def download_file(url: str, target_path: Path):
    import ssl

    if target_path.exists():
        logging.info("File already exists: [%s]", str(target_path.resolve()))
        return

    wget.download(url, str(target_path.resolve()))
