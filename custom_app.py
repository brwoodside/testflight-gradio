import gradio as gr
import together_gradio

gr.load(
    name='meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
    src=together_gradio.registry,
    title='Together-Gradio Integration',
    description="Chat with Meta-Llama-3.1-70B-Instruct-Turbo model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"],
    multimodal=True
).launch()