import gradio as gr
import together_gradio

gr.load(
    name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    src=together_gradio.registry,
    multimodal=True
).launch()