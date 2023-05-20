"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.8
"""

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Splits data into features and targets sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["satisfaction"]
    
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    #regressor = DecisionTreeClassifier()
    regressor = LogisticRegression(solver='lbfgs', max_iter=200)
    #regressor = RandomForestClassifier()
    '''
    model = LogisticRegression(
        C=0.056,
        class_weight={},
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=1000,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=123,
        solver='lbfgs',
        tol=0.0001,verbose=0,
        warm_start=False)
    '''
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) :
    """
    Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)[:,1]
    
    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probas)
    print('ROC AUC: %.3f'%roc_auc)
    print('Accuracy: %.3f'%accuracy)
    
    #printout the results
    logger=logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f on test data.",accuracy)
    logger.info("Model has an ROC AUC of %.3f on test data.",roc_auc)