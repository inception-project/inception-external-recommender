import logging


def setup_logging(level=logging.DEBUG):
    log_fmt = "%(process)d-%(thread)d %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=log_fmt)

    filelock_logger = logging.getLogger("filelock")
    filelock_logger.setLevel(logging.WARNING)
