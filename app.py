import gradio as gr
import pickle

# Load your GBModel
def load_model():
    with open('models/GBModel.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to get model weight
def get_model_weight():
    model = load_model()
    return f"The model weighs: {model.weight} units"

# Function to make predictions (example)
def predict(input_data):
    model = load_model()
    # Replace this with your model's prediction logic
    prediction = model.predict([input_data])
    return f"Prediction: {prediction[0]}"

# Gradio Interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# GBModel Interface")
        
        with gr.Tab("Model Weight"):
            gr.Markdown("Click the button below to see how much the model weighs.")
            weight_button = gr.Button("Get Model Weight")
            weight_output = gr.Textbox(label="Model Weight")
            weight_button.click(get_model_weight, outputs=weight_output)
        
        with gr.Tab("Make a Prediction"):
            gr.Markdown("Enter input data to get a prediction from the model.")
            input_data = gr.Textbox(label="Input Data")
            predict_button = gr.Button("Predict")
            prediction_output = gr.Textbox(label="Prediction")
            predict_button.click(predict, inputs=input_data, outputs=prediction_output)
    
    return demo

# Run the Gradio app
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
