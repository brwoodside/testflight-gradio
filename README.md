# `together_gradio`

is a Python package that makes it very easy for developers to create machine learning apps that are powered by Together's Inference API.

# Installation

1. Clone this repo: `git clone git@github.com:gradio-app/together_gradio.git`
2. Navigate into the folder that you cloned this repo into: `cd together_gradio`
3. Install this package: `pip install -e .`

<!-- ```bash
pip install Together-gradio
``` -->

That's it! 

# Basic Usage

Just like if you were to use the `Together` Client, you should first save your together API token to this environment variable:

```
export TOGETHER_API_KEY=<your token>
```

Then in a Python file, write:

```python
import gradio as gr
import together_gradio

gr.load(
    name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    src=together_gradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio ChatInterface connected to the model on Together!

![ChatInterface](chatinterface.png)

# Customization 

Once you can create a Gradio UI from a Together endpoint, you can customize it by setting your own input and output components, or any other arguments to `gr.ChatInterface`. For example, the screenshot below was generated with:

```py
import gradio as gr
import together_gradio

gr.load(
    name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    src=together_gradio.registry,
    title='Together-Gradio Integration',
    description="Chat with Meta-Llama-3.1-70B-Instruct-Turbo model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()
```
![ChatInterface with customizations](chatinterface_with_customization.png)

# Composition

Or use your loaded Interface within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import together_gradio

with gr.Blocks() as demo:
    with gr.Tab("8B"):
        gr.load('meta-llama/Meta-Llama-3-8B-Instruct-Lite', src=Together_gradio.registry)
    with gr.Tab("70B"):
        gr.load('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', src=Together_gradio.registry)

demo.launch()
```

# Under the Hood

The `Together-gradio` Python library has two dependencies: `Together_cloud_sdk` and `gradio`. It defines a "registry" function `Together_gradio.registry`, which takes in a model name and returns a Gradio app.

# Supported Models in Together API
Currently the available options are: llama3.1-8b, llama3.1-70b

-------

Note: if you are getting a 401 authentication error, then the Together API Client is not able to get the API token from the environment variable. This happened to me as well, in which case save it in your Python session, like this:

```py
import os

os.environ["TOGETHER_API_KEY"] = ...
```
