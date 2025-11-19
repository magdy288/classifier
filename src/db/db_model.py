"""
This module defines the database model for storing cryptocurrency data using SQLAlchemy's ORM.

It includes a base class for declarative models and a specific model for cryptocurrency data
with various financial metrics.
"""
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Float, Integer

from config import db_settings

class Base(DeclarativeBase):
    """Base class for declarative models."""
    pass

class CryptoData(Base):
    """
    SQLAlchemy model for cryptocurrency data.
    
    Attributes:
        timestamp (int): The timestamp of the data point.
        Open (float): The opening price.
        High (float): The highest price.
        Low (float): The lowest price.
        Close (float): The closing price.
        Volume (float): The trading volume.
        CCI (float): Commodity Channel Index.
        CMO (float): Chande Momentum Oscillator.
        Target (float): The target value for prediction.
    """
    __tablename__ = db_settings.crypto_table_name

    timestamp: Mapped[int] = mapped_column(Integer, primary_key=True)
    Open: Mapped[float] = mapped_column(Float())
    High: Mapped[float] = mapped_column(Float())
    Low: Mapped[float] = mapped_column(Float())
    Close: Mapped[float] = mapped_column(Float())
    Volume: Mapped[float] = mapped_column(Float())
    CCI: Mapped[float] = mapped_column(Float())
    CMO: Mapped[float] = mapped_column(Float())
    Target: Mapped[float] = mapped_column(Float())
    