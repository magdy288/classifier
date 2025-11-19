"""
Main runner for the ML model service.

This script initializes the model service, loads the pre-trained model,
collects the necessary data, and makes predictions based on the latest data point.

It demonstrates how to use the ModelService class to perform predictions
on financial data.
"""

from loguru import logger
from warnings import filterwarnings

from model.model_service import ModelService
from model.pipeline.collection import collect_data

filterwarnings('ignore')

symbol = 'BTCUSDT'
timeframe = '5m'

@logger.catch
def main(
    symbol: str , 
    timeframe: str,
):
    """
    Run the ML model service to make predictions.
    
    Loads the model, collects data, and outputs predictions.
    
    Args:
        symbol (str): The trading symbol to analyze.
        timeframe (str): The timeframe for data collection.
    """
    logger.info('Running the application...')
    
    ml_svc = ModelService(
        symbol,
        timeframe,
    )
    ml_svc.load_model()
    test = collect_data(
        symbol,
        timeframe,
    )
    pred = ml_svc.predict([test['High'].values[-1],
                           test['Low'].values[-1],
                           test['Open'].values[-1],
                           test['Volume'].values[-1],
                           test['Close'].values[-1],
                           test['CCI'].values[-1],
                           test['CMO'].values[-1],
                           ])
    
    logger.info(f'Prediction = {pred}')
    print(f'Prediction = {pred}')
    

if __name__ == '__main__':
    main(
        symbol,
        timeframe
    )
