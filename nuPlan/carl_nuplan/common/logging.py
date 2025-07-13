import logging
from functools import wraps


def suppress_info_logs(func):
    """Decorator to suppress INFO logs in the wrapped function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        original_level = logger.level
        logger.setLevel(max(original_level, logging.WARNING))
        try:
            return func(*args, **kwargs)
        finally:
            logger.setLevel(original_level)  # Restore original level

    return wrapper


class SuppressInfoLogs:
    """Context manager to suppress INFO logs within a block of code."""

    def __enter__(self):
        self.logger = logging.getLogger()
        self.original_level = self.logger.level
        self.logger.setLevel(max(self.original_level, logging.WARNING))

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.setLevel(self.original_level)
