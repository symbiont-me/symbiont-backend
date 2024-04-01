import logging
from colorlog import ColoredFormatter


class Logger:
    def __init__(self, logger_name, log_file=None, environment=None):
        # Create a logger with the specified name
        self.logger = logging.getLogger(logger_name)

        # Set logger level based on the environment
        if environment and environment.lower() == "production":
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.DEBUG)

        LOGFORMAT = "%(log_color)s%(asctime)-8s%(reset)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
        # Create a formatter
        color_formatter = ColoredFormatter(LOGFORMAT)

        # Create a console handler and set the formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)
        self.logger.addHandler(console_handler)

        # If a log file is specified, create a file handler and set the formatter
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_format = (
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
            )
            file_formatter = logging.Formatter(file_format)

            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


# Usage:
# environment = os.getenv("ENV")  # or pass the environment directly
# logger_instance = Logger(__name__, log_file="my_log.log", environment=environment)
# logger_instance.debug("This is a debug message")
# logger_instance.info("This is an info message")
# logger_instance.warning("This is a warning message")
# logger_instance.error("This is an error message")
# logger_instance.critical("This is a critical message")
