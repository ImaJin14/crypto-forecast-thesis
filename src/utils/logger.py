"""Structured logging using loguru."""
import sys
from loguru import logger


def setup_logger(level: str = "INFO", log_file: str = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
                      "<level>{message}</level>")
    if log_file:
        logger.add(log_file, rotation="10 MB", retention="30 days", level=level)


def get_logger(name: str = __name__):
    return logger.bind(module=name)
