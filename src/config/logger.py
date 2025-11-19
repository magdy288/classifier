"""
This module configures the logging settings for the application using Loguru.

It reads the log level from environment variables defined in a .env file
and sets up a log file with daily rotation and retention policies.
"""

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

class LoggerSettings(BaseSettings):
    """
    Logger configuration settings loaded from environment variables.
    
    Attributes:
        model_config: Configuration for Pydantic settings, specifying the .env file location and encoding.
        log_level: The logging level (e.g., DEBUG, INFO, WARNING, ERROR), read from the environment.
    """
    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    log_level: str

def configure_logger(log_level: str):
    """
    Configures the Loguru logger with specified settings.
    
    Args:
        log_level (str): The logging level to set for the logger.
    
    Returns:
        None
    """
    logger.remove()  # Remove default logger
    logger.add(
        'logs/app.log',
        rotation='1 day',
        retention='2 days',
        compression='zip',
        level=log_level
    )

configure_logger(log_level='INFO')