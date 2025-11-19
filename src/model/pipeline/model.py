"""
This module contains functions to build, train, evaluate, and save a machine learning model
pipeline using a Random Forest Classifier.

It includes data collection, feature-target separation, train-test splitting,
hyperparameter tuning with GridSearchCV, model evaluation, and model persistence.
"""
import pickle as pk
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from loguru import logger

from model.pipeline.collection import collect_data
from config import model_settings

def build_model(
    symbol: str,
    timeframe: str,
    ) -> None:
    """
    Builds, trains, evaluates, and saves a Random Forest Classifier model pipeline.
    
    This function orchestrates the entire process from data collection to model persistence.
    It's starts by collecting data for the specified symbol and timeframe,
    then separates features and target variables, splits the data into training and testing sets,
    trains a Random Forest Classifier with hyperparameter tuning using GridSearchCV,
    evaluates the model's performance, and finally saves the trained model to disk.
    
    Args:
        symbol (str): The financial symbol for which to collect data.
        timeframe (str): The timeframe for the data collection.
    """
    logger.info('Building a machine learning model pipeline')
    
    df = collect_data(
        symbol,
        timeframe
    )
    
    features = [
        'High',
        'Low',
        'Open',
        'Volume',
        'Close',
        'CCI',
        'CMO'
    ]
    X, y = _get_x_y(
        df,
        col_x=features,
        )
    
    X_train, X_test, y_train, y_test = _split_train_test(
        X,
        y
    )
    
    rf = _train_model(
        X_train,
        y_train
    )
    
    _evaluate_model(
        rf,
        X_test,
        y_test
    )
    
    _save_model(rf)
    
    

def _get_x_y(
    data: pd.DataFrame,
    col_x: list[str],
    col_y: str = 'Target'
    ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates features and target variable from the dataset.
    
    Args:
        data (pd.DataFrame): The input dataset containing features and target.
        col_x (list[str]): List of column names to be used as features.
        col_y (str): The column name to be used as the target variable. Default is 'Target'.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and target Series.
    """
    logger.info(f'defining X and Y variables. \nX vars: {col_x}\ny var: {col_y}')
    return data[col_x], data[col_y]

def _split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.
    
    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target variable Series.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing
        the training features, testing features, training target, and testing target.
    """
    logger.info('Splitting data into train and test sets')
    
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    ) # type: ignore

def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Trains a Random Forest Classifier model with hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): The training features DataFrame.
        y_train (pd.Series): The training target Series.
    
    Returns:
        RandomForestClassifier: The best estimator after GridSearch.
    """
    logger.info('Training a model with hyperparameters')
    
    params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 6, 9, 12],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid=params,
        cv=5,
    )
    
    model_grid = grid_search.fit(
        X_train,
        y_train
    )
    
    return model_grid.best_estimator_


def _evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """
    Evaluates the trained model's performance on the test set.
    
    Args:
        model (RandomForestClassifier): The trained Random Forest Classifier model.
        X_test (pd.DataFrame): The testing features DataFrame.
        y_test (pd.Series): The testing target Series.
    
    Returns:
        float: The accuracy score of the model on the test set.
    """
    model_score = model.score(X_test, y_test)
    
    logger.info(f'Evaluating model performance. SCORE={model_score}')
    
    return model_score # type: ignore

def _save_model(model: RandomForestClassifier) -> None:
    """
    Saves the trained model to disk using pickle to the specified path in model settings directory.
    
    Args:
        model (RandomForestClassifier): The trained Random Forest Classifier model to be saved.
    """
    
    model_path = f'{model_settings.model_path}/{model_settings.model_name}'
    logger.info(f'Saving a model to a directory: {model_path}')
    
    with open(model_path, 'wb') as f:
        pk.dump(model, f)    
