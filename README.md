# `testflight_gradio`

is a Python package that makes it very easy for developers to create machine learning apps that are powered by Positron's Testflight Inference API.

# Installation

1. Clone this repo: `git clone git@github.com:gradio-app/testflight_gradio.git`
2. Navigate into the folder that you cloned this repo into: `cd testflight_gradio`
3. Install this package: `pip install -e .`

<!-- ```bash
pip install testflight-gradio
``` -->

That's it! 

# Basic Usage

Just like if you were to use the `testflight` Client, you should first save your Testflight API token to this environment variable:

```sh
export TESTFLIGHT_API_KEY=<your token>
```

Then in a Python file, write:

```python
import gradio as gr
import testflight_gradio

gr.load(
    name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    src=testflight_gradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio ChatInterface connected to the model on testflight!

![ChatInterface](chatinterface.png)

# Customization 

Once you can create a Gradio UI from a testflight endpoint, you can customize it by setting your own input and output components, or any other arguments to `gr.ChatInterface`. For example, the screenshot below was generated with:

```py
import gradio as gr
import testflight_gradio

gr.load(
    name='llama-3.1-70b-instruct-good-tp2',
    src=testflight_gradio.registry,
    title='testflight-Gradio Integration',
    description="Chat with Meta-Llama-3.1-70B-Instruct-Turbo model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()
```
![ChatInterface with customizations](chat_custom.png)

# Composition

Or use your loaded Interface within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import testflight_gradio

with gr.Blocks() as demo:
    with gr.Tab("8B"):
        gr.load('llama-3.1-8b-instruct-good-tp2', src=testflight_gradio.registry)
    with gr.Tab("70B"):
        gr.load('llama-3.1-70b-instruct-good-tp2', src=testflight_gradio.registry)

demo.launch()
```

# Under the Hood

The `testflight-gradio` Python library has two dependencies: `openai` and `gradio`. It defines a "registry" function `testflight_gradio.registry`, which takes in a model name and returns a Gradio app.

# Supported Models in testflight API
Currently the available options are: llama3.1-8b, llama3.1-70b

-------

Note: if you are getting a 401 authentication error, then the testflight API Client is not able to get the API token from the environment variable. This happened to me as well, in which case save it in your Python session, like this:

```py
import os

os.environ["TESTFLIGHT_API_KEY"] = ...
```
