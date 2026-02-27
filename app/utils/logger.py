"""
Logger configuration using loguru.
Easy to use, effective, and prepared for deployment.

Installation: pip install loguru
"""

from loguru import logger
import sys

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

# Add file handler for persistent logs (.log format)
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # New file every day at midnight
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress old logs
    level="INFO"
)

# Add text file handler (.txt format, without color codes)
logger.add(
    "logs/app_{time:YYYY-MM-DD}.txt",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="00:00",
    retention="30 days",
    level="INFO",
    colorize=False  # Remove color codes for plain text
)

# Export logger instance
__all__ = ["logger"]
