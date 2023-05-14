import logging
import os

# Define ANSI escape sequences for colors
COLORS = {
    "DEFAULT": "\033[0m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "CYAN": "\033[96m",
    "BOLD_RED": "\033[1;31m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_level = record.levelname
        if log_level == "INFO":
            color_code = "\033[92m"  # Green
        elif log_level == "DEBUG":
            color_code = "\033[93m"  # Yellow
        elif log_level == "ERROR":
            color_code = "\033[91m"  # Red
        else:
            color_code = "\033[0m"  # Default (no color)

        log_msg = super().format(record)
        return f"{color_code}{log_msg}\033[0m"


class CustomLogger:
    def __init__(self, log_file, create_directory=False, log_level=logging.INFO):
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(log_level)

        if create_directory:
            log_directory = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_directory, exist_ok=True)
            log_path = os.path.join(log_directory, log_file)
            handler = logging.FileHandler(log_path)
        else:
            handler = logging.StreamHandler()

        formatter = (
            ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
            if not create_directory
            else logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(message)

    def exception(self, message):
        self.logger.exception(message)

    def debug(self, message):
        self.logger.debug(message)
