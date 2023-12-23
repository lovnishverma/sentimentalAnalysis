import gradio as gr
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(input_text):
    summary = summarizer(input_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']

iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    live=True,
    title="Text Summarization App",
    description="Enter a longer document, and the model will generate a summary.",
)

iface.launch()
