import logging


def setup_logging(log_file):
    log_formatter = logging.Formatter(
        fmt=('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  '
             '%(message)s'),
        datefmt='%m/%d/%Y %H:%M:%S %p')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # pil_logger = logging.getLogger('PIL')
    # pil_logger.setLevel(logging.INFO)
    return logger
