import gradio as gr
from transformers import pipeline
sentiment_analyzer = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')
def analyze_sentiment(text):
    candidate_labels = ["positive", "negative"]
    result = sentiment_analyzer(text, candidate_labels)
    return result['labels'][0], result['scores'][0]
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter Text",placeholder="Enter your text here...."),
    outputs=[gr.Text(label="Sentiment Output"), gr.Number(label="Confidence Score")],
    live=False, 
    title="Sentiment Analysis with BART",
    description="Enter a text to analyze its sentiment (positive/negative) using the BART model.",
    theme="compact",
)


iface.launch()
