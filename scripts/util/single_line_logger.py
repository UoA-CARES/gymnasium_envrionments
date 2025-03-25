import logging
import sys
import time


class SingleLineLogger(logging.StreamHandler):
    """Custom handler to overwrite the same terminal line for progress updates."""

    def emit(self, record):
        msg = self.format(record)
        sys.stdout.write(f"\r{msg}   ")  # Overwrite the line with some padding
        sys.stdout.flush()


def log_progress(logger, message, standard_handler):
    """
    Logs progress by overwriting the current terminal line.

    Parameters:
        logger (logging.Logger): The logger to use.
        message (str): The message to log.
        standard_handler (logging.Handler): The default logging handler.
    """
    temp_handler = SingleLineLogger()
    temp_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(temp_handler)  # Attach the single-line handler
    logger.removeHandler(standard_handler)  # Temporarily remove standard logging
    logger.propagate = False  # Prevent double logging
    logger.info(message)  # Log progress (overwrites previous message)
    logger.propagate = True  # Restore logging propagation
    logger.removeHandler(temp_handler)  # Remove single-line handler
    logger.addHandler(standard_handler)  # Restore standard logging
