import gradio as gr
import testflight_gradio

gr.load(
    name='llama-3.1-70b-instruct-good-tp2',
    src=testflight_gradio.registry,
    multimodal=False
).launch()