import os
import base64
from openai import OpenAI
import gradio as gr
from typing import Callable

__version__ = "0.0.1"   

def get_image_base64(url: str, ext: str):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return "data:image/" + ext + ";base64," + encoded_string

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    def fn(message, history):
        inputs = preprocess(message, history)
        client = OpenAI(
            base_url="https://testflight-sandworm-2.fly.dev/api/v1",
            api_key=api_key,
        )
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=inputs["messages"],
                stream=True,
            )
            response_text = ""
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                response_text += delta
                yield postprocess(response_text)
        except Exception as e:
            error_message = f"Error: {str(e)}"
            return error_message

    return fn

def handle_user_msg(message: str):
    if type(message) is str:
        return message
    else:
        raise NotImplementedError

def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):           
            messages = []
            files = None
            for user_msg, assistant_msg in history:
                if assistant_msg is not None:
                    messages.append({"role": "user", "content": handle_user_msg(user_msg)})
                    messages.append({"role": "assistant", "content": assistant_msg})
                else:
                    files = user_msg
            if type(message) is str and files is not None:
                message = {"text":message, "files":files}
            elif type(message) is dict and files is not None:
                if message["files"] is None or len(message["files"]) == 0:
                    message["files"] = files
            messages.append({"role": "user", "content": handle_user_msg(message)})
            return {"messages": messages}

        postprocess = lambda x: x  # No post-processing needed
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    # Determine the pipeline type based on the model name
    # For simplicity, assuming all models are chat models at the moment
    return "chat"


def registry(name: str, token: str | None = None, **kwargs):
    """
    Create a Gradio Interface for a model on testflight.

    Parameters:
        - name (str): The name of the model on testflight.
        - token (str, optional): The API key for testflight.
    """
    api_key = token or os.environ.get("TESTFLIGHT_API_KEY")
    if not api_key:
        raise ValueError("TESTFLIGHT_API_KEY environment variable is not set.")

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(name, preprocess, postprocess, api_key)

    if pipeline == "chat":
        interface = gr.ChatInterface(fn=fn, **kwargs)
    else:
        # For other pipelines, create a standard Interface (not implemented yet)
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface