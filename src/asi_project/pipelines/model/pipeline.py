"""
This is a boilerplate pipeline 'linear_regression'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import split_data, train_model, evaluate_model, create_gradio


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_flights_train", "params:features"],
                outputs=["X_train", "y_train"],
                name="split_training_data_node",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_flights_test", "params:features"],
                outputs=["X_test", "y_test"],
                name="split_testing_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier", "X_test", "y_test", "params:model_options"],
                outputs="dummy",
                name="evaluate_model_node",
            ),
            node(
                func=create_gradio,
                inputs=["params:features", "dummy"],
                outputs=None,
                name="create_gradio_node",
            ),
        ],
        namespace = "model",
        inputs = ["preprocessed_flights_train", "preprocessed_flights_test"],
        outputs = "classifier")
