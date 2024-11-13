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
    def fn(message: Union[str, Dict[str, Any]], history: list) -> str:
        """
        Process chat messages and return response.
        Args:
            message: Either a string (text-only) or dict with 'text' and 'files' keys (multimodal)
            history: List of messages with 'role' and 'content' keys
        Returns:
            str: The model's response
        """
        try:
            inputs = preprocess(message, history)
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
            )
            
            logging.info(f"Sending request to model: {model_name}")
            
            # Format messages for Together AI API
            api_messages = []
            for msg in inputs["messages"]:
                if isinstance(msg["content"], list):
                    # Handle multimodal content
                    content = []
                    for item in msg["content"]:
                        if isinstance(item, dict) and "path" in item:
                            # Convert image path to base64
                            ext = os.path.splitext(item["path"])[1].strip(".")
                            encoded_str = get_image_base64(item["path"], ext)
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": encoded_str}
                            })
                        else:
                            content.append({
                                "type": "text",
                                "text": str(item)
                            })
                    api_messages.append({
                        "role": msg["role"],
                        "content": content
                    })
                else:
                    # Handle text-only content
                    api_messages.append({
                        "role": msg["role"],
                        "content": str(msg["content"])
                    })
            
            logging.debug(f"API messages: {api_messages}")
            
            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                stream=True,
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
            
            # Add a system message
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Answer questions clearly and concisely."
            })
            
            # Process history (messages format)
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Process current message
            if isinstance(message, dict):
                content = []
                # Add text if present
                if message.get("text"):
                    content.append(message["text"])
                
                # Add images if present
                if message.get("files"):
                    for file_path in message["files"]:
                        content.append({"path": file_path})
                
                # For API request
                api_content = handle_user_msg(message)
                messages.append({
                    "role": "user",
                    "content": api_content
                })
                
                # For display in chat
                if content:
                    messages.append({
                        "role": "user",
                        "content": content
                    })
            else:
                content = str(message)
                messages.append({
                    "role": "user",
                    "content": content
                })
            
            logging.debug(f"Preprocessed messages: {messages}")
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