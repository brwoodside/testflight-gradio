import gradio as gr
import testflight_gradio

gr.load(
    name='llama-3.1-70b-instruct-good-tp2',
    src=testflight_gradio.registry,
    title='Positron Testflight-Gradio Integration',
    description="Chat with Meta-Llama-3.1-70B-Instruct model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"],
    multimodal=False
).launch()