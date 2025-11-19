"""
This module defines the configuration settings for the machine learning model,
including the model's file path and name.

It uses Pydantic's BaseSettings to load settings from an environment file.
"""
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelSettings(BaseSettings):
    """
    ML model configuration settings.
    
    Attributes:
        model_config (SettingsConfigDict): Configuration for loading settings from an env file.
        model_path (DirectoryPath): The file system path to the ML model.
        model_name (str): The name of the ML model.
    """
    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    model_path : DirectoryPath
    model_name : str


model_settings = ModelSettings()