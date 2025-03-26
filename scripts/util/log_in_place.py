import logging
import sys


class InPlaceLogger(logging.Logger):
    """Custom logger that immediately uses in-place logging handler for same line logging."""

    def __init__(self, name: str):
        super().__init__(name, level=logging.INFO)
        self.addHandler(LogInPlaceHandler())
        self.propagate = False


class LogInPlaceHandler(logging.StreamHandler):
    """Custom handler to overwrite the same terminal line for progress updates."""

    def __init__(self):
        super().__init__(stream=sys.stdout)
        formatter = logging.Formatter("%(levelname)s:%(message)s")
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        sys.stdout.write(f"{msg}    \r")  # Overwrite the line with some padding
        sys.stdout.flush()
