'''Simple demo gradio interface - not for main app, demo only

# existing local venv: source ../3-simple-chatbot/.venv/bin/activate
# for new env: python -m venv .venv ; source .venv/bin/activate

# installing required libraries in my_env
pip install transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1
'''

import gradio as gr
# from gradio.flagging import CSVLogger - not need to be explicit


# logger = CSVLogger()  # writes a CSV + any files - not need to be explicit

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(
    fn=greet, 
    inputs="text", 
    outputs="text", 
    # flagging_dir="flagged",                  # where to save - not need to be explicit
    # flagging_options=["bug", "incorrect"],   # optional reasons dropdown
    # flagging_callback=logger)               # actually log to CSV) - not need to be explicit
)

demo.launch(server_name="0.0.0.0", server_port= 7860)

'''
The above code creates a gradio.Interface called demo. It wraps the greet function with a simple text-to-text user interface that you could interact with.

The gradio.Interface class is initialized with 3 required parameters:

fn: the function to wrap a UI around
inputs: which component(s) to use for the input (e.g. “text”, “image” or “audio”)
outputs: which component(s) to use for the output (e.g. “text”, “image” or “label”)
The last line demo.launch() launches a server to serve our demo.

RUN IN TERMINAL: python3 hello.py
'''