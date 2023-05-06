import os, logging


class CustomLogger:
    def __init__(self, log_file):
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(os.path.join(os.getcwd(), "logs", log_file))
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(message)

    def exception(self, message):
        self.logger.exception(message)
