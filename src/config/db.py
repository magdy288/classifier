"""
This module sets up the database connection using SQLAlchemy and loads configuration
settings from environment variables using Pydantic.

It defines a DBSettings class to manage database-related settings and creates.

"""
from sqlalchemy import create_engine
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    """
    Database configuration settings for the application.
    
    Attributes:
        model_config: Configuration for Pydantic settings, including the path to the .env file.
        crypto_database_url (str): The database connection URL.
        crypto_table_name (str): The name of the table used for cryptocurrency data.
    """
    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    crypto_database_url : str
    crypto_table_name : str

db_settings = DBSettings()

engine = create_engine(db_settings.crypto_database_url)