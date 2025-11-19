"""
This module contains the ModelService class responsible for loading
and utilizing a machine learning model for predictions.

It checks for the existence of a pre-trained model file, builds
the model if it doesn't exist, and provides a method to make predictions
using the loaded model.
"""
from pathlib import Path
import pickle as pk

from loguru import logger

from model.pipeline.model import build_model
from config import model_settings

class ModelService:
    """
    A service class for loading and using a machine learning model
    for making predictions.
    
    This class checks for the existence of a pre-trained model file,
    builds the model if it doesn't exist, and provides a method to
    make predictions using the loaded model.
    
    Attributes:
        model: The loaded machine learning model.
        symbol (str): The financial symbol for which the model is built.
        timeframe (str): The timeframe for which the model is built.
    
    Methods:
        load_model(): Loads the machine learning model from a file, 
            building it if it does not exist.
        predict(input_parameters: list) -> list: Makes a prediction
            using the loaded model based on the provided input parameters.
        
    """
    def __init__(self, symbol: str, timeframe: str):
        """ Initializes the ModelService with the specified symbol and timeframe."""
        self.model = None
        self.symbol = symbol
        self.timeframe = timeframe
        
    def load_model(self):
        """
        Loads the machine learning model from a file. If the model file does not exist,
        it builds the model using the specified symbol and timeframe.
        """
        logger.info(
            'Checking the existance of model config file at '
            f'{model_settings.model_path}/{model_settings.model_name}'
        )
        
        model_path = Path(
            f'{model_settings.model_path}/{model_settings.model_name}'
        )
        
        if not model_path.exists():
            logger.warning(
                f'Model at {model_path} was not found -> '
                f'BUILDING {model_settings.model_name}'
            )
            build_model(self.symbol, self.timeframe)
        
        logger.info(
            f'Model {model_settings.model_name} exists -> '
            'loading model configuration file'
        )
        
        with open(model_path, 'rb') as model_file:
            self.model = pk.load(model_file)        
    
    def predict(self, input_parameters: list) -> list:
        """
        Makes a prediction using the loaded model based on the provided input parameters.
        
        Takes a list of input parameters and returns the model's prediction as a list.
        
        Args:
            input_parameters (list): A list of input parameters for the model.
        
        Returns:
            list: The model's prediction.
        """
        logger.info('Making prediction')
        return self.model.predict([input_parameters]) # type: ignore
        
    