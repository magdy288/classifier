"""
This module is responsible for collecting cryptocurrency data from an exchange,
computing technical indicators, and storing the data in a database.

It uses the ccxt library to interact with the exchange API, pandas for data manipulation,
and SQLAlchemy for database operations.

And it includes functions to calculate technical indicators like CCI and CMO.
"""
import pandas as pd
import ccxt
import numpy as np

from loguru import logger
from sqlalchemy import select

from config import engine
from db.db_model import CryptoData

def collect_data(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1m', 
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Extracts OHLCV data from Binance, computes CCI and CMO indicators,
    and stores the data in the database.
    
    Args:
        symbol (str): The trading pair symbol to fetch data for.
        timeframe (str): The timeframe for the OHLCV data.
        limit (int): The number of data points to fetch.
    
    Returns:
        pd.DataFrame: A DataFrame containing the collected and processed data.
    """
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    df['CCI'] = _calculate_cci(df, period=14)
    df['CMO'] = _calculate_cmo(df['Close'], period=14)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    df = df.dropna()
    df.to_sql('crypto', con=engine, if_exists='replace', index=False)
    
    query = select(CryptoData)
    
    data = pd.read_sql(query, con=engine)
    logger.info(f'Collected {len(data)} rows of data for {symbol} at {timeframe} timeframe.')
    return data



def _calculate_cci(
    df: pd.DataFrame,
    period: int =14
) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI) for a given DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        period (int): Number of periods for the moving average and mean deviation (default 14).
    
    Returns:
        pd.Series: CCI values.
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    
    sma_tp = tp.rolling(window=period).mean()
    
    mad = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    
    cci = (tp - sma_tp) / (0.015 * mad)
    
    return cci


def _calculate_cmo(
    prices: pd.Series,
    period: int =14,
) -> pd.Series:
    """
    Calculate the Chande Momentum Oscillator (CMO).
    
    Args:
        prices (pd.Series): Series of closing prices.
        period (int): The lookback period for the calculation. (Default is 14).
    
    Returns:
        pd.Series: The CMO values.
    """
    changes = prices.diff()
    
    gains = changes.where(changes > 0, 0)
    losses = -changes.where(changes < 0, 0)
    
    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()
    
    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    
    cmo = cmo.replace([np.inf, -np.inf], np.nan)
    
    return cmo
