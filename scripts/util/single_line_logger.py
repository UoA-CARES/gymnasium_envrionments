import logging
import sys

class SingleLineLogger(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        self.setFormatter(formatter)

    """Custom handler to overwrite the same terminal line for progress updates."""

    def emit(self, record):
        msg = self.format(record)
        sys.stdout.write(f"\r{msg}   ")  # Overwrite the line with some padding
        sys.stdout.flush()
