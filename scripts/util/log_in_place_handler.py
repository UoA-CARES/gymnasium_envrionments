import logging
import sys


class LogInPlaceHandler(logging.StreamHandler):
    """Custom handler to overwrite the same terminal line for progress updates."""

    def __init__(self):
        super().__init__(stream=sys.stdout)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        sys.stdout.write(f"\r{msg}\t")  # Overwrite the line with some padding
        sys.stdout.flush()

    def log(self, logger: logging.Logger, msg: str) -> None:
        logger.propagate = False
        logger.info(msg)
        logger.propagate = True
