import gradio as gr
from transformers import pipeline

# Load pre-trained sentiment analysis model
classifier = pipeline('sentiment-analysis')

# Define the sentiment analysis function
def analyze_sentiment(text):
    result = classifier(text)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    return f"Sentiment: {sentiment}, Confidence: {confidence:.4f}"

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    live=True,
    title="Sentiment Analysis App",
    description="Enter a sentence, and the model will predict its sentiment.",
)

# Launch the Gradio app
iface.launch()
