# Setup and run gradio
from kedro_mlflow.io.models import MlflowModelLoggerDataSet
import gradio as gr


def setup_gradio(features, model):
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