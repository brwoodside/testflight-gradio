import os
import base64
import logging
from openai import OpenAI
import gradio as gr
from typing import Callable, Dict, Any, Union
from PIL import Image
import io

__version__ = "0.0.1"

logging.basicConfig(level=logging.INFO)

def resize_image(image_path: str, max_size: int = 768) -> str:
    """Resize image to limit dimensions while maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Calculate new dimensions
            ratio = max_size / max(img.size)
            if ratio < 1:  # Only resize if image is larger than max_size
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return "data:image/jpeg;base64," + encoded_string
            
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return None

def get_image_base64(url: str, ext: str) -> str:
    """Get base64 string of image, with resizing if needed."""
    return resize_image(url) or "data:image/" + ext + ";base64," + base64.b64encode(open(url, "rb").read()).decode('utf-8')

def format_multimodal_message(message: Dict[str, Any]) -> list:
    """Format a multimodal message for the API."""
    content = []
    
    # Add text content if present
    if message.get("text"):
        content.append({
            "type": "text",
            "text": message["text"]
        })
    
    # Add image content if present
    if message.get("files"):
        for file_path in message["files"]:
            ext = os.path.splitext(file_path)[1].strip(".")
            if ext.lower() in ["png", "jpg", "jpeg", "gif"]:
                encoded_str = get_image_base64(file_path, ext)
                if encoded_str:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": encoded_str}
                    })
            else:
                logging.warning(f"Unsupported file type: {ext}")
    
    return content

def handle_user_msg(message: Union[str, Dict[str, Any]]) -> Union[str, list]:
    """Handle different types of user messages."""
    if isinstance(message, str):
        return message
    elif isinstance(message, dict):
        if message.get("files"):
            return format_multimodal_message(message)
        return message.get("text", "")
    else:
        raise NotImplementedError(f"Unsupported message type: {type(message)}")

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    def fn(message, history):
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
            )
            
            inputs = preprocess(message, history)
            
            stream = client.chat.completions.create(
                model=model_name,
                messages=inputs["messages"],
                stream=True,
                max_tokens=1024,
                temperature=0.7,
            )
            
            response_text = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    yield postprocess(response_text)
                
        except Exception as e:
            error_message = f"Error: {str(e)}"
            logging.error(f"Error in chat completion: {error_message}")
            yield error_message

    return fn

def get_interface_args(pipeline):
    if pipeline == "chat":
        def preprocess(message, history):
            messages = []
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Answer questions clearly and concisely."
            })
            
            # Process history
            files = None
            for user_msg, assistant_msg in history:
                if assistant_msg is not None:
                    # Handle user message
                    if isinstance(user_msg, dict) and user_msg.get("files"):
                        content = []
                        if user_msg.get("text"):
                            content.append({
                                "type": "text",
                                "text": user_msg["text"]
                            })
                        for file_path in user_msg["files"]:
                            ext = os.path.splitext(file_path)[1].strip(".")
                            if ext.lower() in ["png", "jpg", "jpeg", "gif"]:
                                encoded_str = get_image_base64(file_path, ext)
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": encoded_str}
                                })
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "user", "content": str(user_msg)})
                    
                    # Add assistant message
                    messages.append({"role": "assistant", "content": assistant_msg})
                else:
                    files = user_msg["files"] if isinstance(user_msg, dict) else None

            # Handle current message
            if isinstance(message, str) and files:
                message = {"text": message, "files": files}
            elif isinstance(message, dict) and files and not message.get("files"):
                message["files"] = files

            # Format current message
            if isinstance(message, dict) and message.get("files"):
                content = []
                if message.get("text"):
                    content.append({
                        "type": "text",
                        "text": message["text"]
                    })
                for file_path in message["files"]:
                    ext = os.path.splitext(file_path)[1].strip(".")
                    if ext.lower() in ["png", "jpg", "jpeg", "gif"]:
                        encoded_str = get_image_base64(file_path, ext)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": encoded_str}
                        })
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": str(message)})

            return {"messages": messages}

        def postprocess(x: str) -> str:
            return x

        return None, None, preprocess, postprocess
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline}")

def get_pipeline(model_name):
    return "chat"

def registry(name: str, token: str | None = None, **kwargs):
    """
    Create a Gradio Interface for a model on Together AI.

    Parameters:
        - name (str): The name of the model on Together AI.
        - token (str, optional): The API key for Together AI.
        - **kwargs: Additional arguments to pass to the ChatInterface
    """
    api_key = token or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is not set.")

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(name, preprocess, postprocess, api_key)

    if pipeline == "chat":
        # Set default kwargs for chat interface
        chat_kwargs = {
            "fn": fn,
            "type": "messages",
            "multimodal": True,  # Enable multimodal support
        }
        # Update with any additional kwargs
        chat_kwargs.update(kwargs)
        
        interface = gr.ChatInterface(**chat_kwargs)
    else:
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface