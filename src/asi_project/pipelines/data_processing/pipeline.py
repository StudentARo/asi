"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_flights


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=preprocess_flights,
                inputs="flights_train",
                outputs="preprocessed_flights_train",
                name="preprocess_flights_train_node",
            ),
            node(
                func=preprocess_flights,
                inputs="flights_test",
                outputs="preprocessed_flights_test",
                name="preprocess_flights_test_node",
            ),
    ])
