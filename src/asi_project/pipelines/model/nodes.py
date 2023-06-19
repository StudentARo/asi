"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.8
"""

import logging
from typing import Dict, Tuple
import gradio as gr

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kedro_mlflow.io.models import MlflowModelLoggerDataSet

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def split_data(data: pd.DataFrame, features) -> Tuple:
    """
    Splits data into features and targets sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[features]
    y = data["satisfaction"]
    
    return X, y

createRegressor = {"LogisticRegression": lambda model_options: LogisticRegression(solver=model_options["solver"], max_iter=model_options["iterations"]),
                   "RandomForestClassifier": lambda model_options: RandomForestClassifier(max_depth=model_options["max_depth"]),
                   "DecisionTreeClassifier": lambda model_options: DecisionTreeClassifier(max_depth=model_options["max_depth"])}

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_options: Dict) -> LinearRegression:
    """
    Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = createRegressor[model_options["algorithm"]](model_options)
    regressor.fit(X_train, y_train)
    
    mlflow_model_logger = MlflowModelLoggerDataSet(flavor="mlflow.sklearn")
    mlflow_model_logger.save(regressor)
    return regressor


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) :
    """
    Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    mlflow_model_logger = MlflowModelLoggerDataSet(flavor="mlflow.sklearn")
    model = mlflow_model_logger.load()
    
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)[:,1]
    signature = infer_signature(X_test, y_pred)
    
    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="Flights",
    )
    
    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probas)
    #r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("accuracy_score", accuracy)
    mlflow.log_metric("roc_auc_score", roc_auc)
    #mlflow.log_metric("r2_score", r2)
    print('ROC AUC: %.3f'%roc_auc)
    print('Accuracy: %.3f'%accuracy)
    
    # log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    t_n, f_p, f_n, t_p = cm.ravel()
    mlflow.log_metric("tn", t_n)
    mlflow.log_metric("fp", f_p)
    mlflow.log_metric("fn", f_n)
    mlflow.log_metric("tp", t_p)
    
    #printout the results
    logger=logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f on test data.",accuracy)
    logger.info("Model has an ROC AUC of %.3f on test data.",roc_auc)
    
    
def create_gradio(features, model):
    with gr.Blocks() as demo:
        with gr.Row():
            input_component = gr.DataFrame(headers=features, row_count=1, label="Input Data", interactive=True)
        with gr.Row():
            predict_button = gr.Button("Predict")
        with gr.Row():
            output_component = gr.Textbox(label="Output Data")
        predict_button.click(classify, inputs=input_component, outputs=output_component)
    #demo = gr.Interface(fn=classify, inputs=input_component, outputs=output_component, live=True)
    demo.launch()
        
def classify(data):
    mlflow_model_logger = MlflowModelLoggerDataSet(flavor="mlflow.sklearn")
    model = mlflow_model_logger.load()
    if (model.predict(data)[0]):
        return "Satisfied"
    else:
        return "Neutral or dissatisfied"