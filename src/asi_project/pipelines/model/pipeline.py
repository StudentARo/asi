"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_flights_train", "params:model_options"],
                outputs=["X_train", "y_train"],
                name="split_training_data_node",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_flights_test", "params:model_options"],
                outputs=["X_test", "y_test"],
                name="split_testing_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),])
