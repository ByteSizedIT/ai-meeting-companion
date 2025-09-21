'''
- set up a language model (LLM) instance, which could be IBM WatsonxLLM, HuggingFaceHub, or an OpenAI model. 
- establish a prompt template. These templates are structured guides to generate prompts for language models, aiding in output organization (ref https://python.langchain.com/docs/how_to/#prompt-templates)
- develop a transcription function that employs the OpenAI Whisper model to convert speech-to-text - takes an audio file uploaded through a Gradio app interface (preferably in .mp3 format). 
- feed transcribed text into an LLMChain, which integrates the text with the prompt template and forwards it to the chosen LLM. 
- display final output from the LLM is in the Gradio app's output textbox.
'''

'''
# existing local venv: source ../3-simple-chatbot/.venv/bin/activate
# for new env: python -m venv .venv ; source .venv/bin/activate

# installing required libraries in .venv
pip install transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1

pip install python-dotenv
'''

import os
from dotenv import load_dotenv

import torch
import os
import gradio as gr
#from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

#######------------- LLM-------------####

# initiate LLM instance, this can be IBM WatsonX, huggingface, or OpenAI instance

load_dotenv()  # loads .env if present

my_credentials = {
    # "url"    : "https://us-south.ml.cloud.ibm.com",
    "url": os.getenv("IBM_WATSON_URL", "https://us-south.ml.cloud.ibm.com"),
    "apikey": os.environ["IBM_WATSON_APIKEY"]   # Line / Own details  added
}

params = {
        GenParams.MAX_NEW_TOKENS: 1000, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]  # Line / Own details added

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-3-2-11b-vision-instruct',         credentials=my_credentials,
        params=params,
        # project_id="skills-network", 
        project_id=PROJECT_ID   # Own details  added
        )

llm = WatsonxLLM(LLAMA2_model) 

#######------------- Prompt Template-------------####

# This template is structured based on LLAMA2. 
# The special tags are:
#   <s>          : start-of-sequence token (some Llama 2 chat variants expect it)
#   <<SYS>> ... <</SYS>> : "system" section that sets global behavior/instructions
#   [INST] ... [/INST]   : "user instruction" section containing the actual prompt
# If you are using other LLMs, feel free to remove the tags

temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

# here is the simplified version of the prompt template
# temp = """
# List the key points with details from the context: 
# The context : {context} 
# """

# At runtime you'll pass context=<your_text> and LangChain will render the {context} placeholder inside the template string.
pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- Speech2text-------------####

def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline``
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    
    # Transcribe the audio file and return the result
    transcript_txt = pipe(audio_file, batch_size=8)["text"]

    # run the chain to merge transcript text with the template and send it to the LLM
    # Because your PromptTemplate has exactly one input variable ("context"), LLMChain.run(...) lets you pass either:
    # a single string (it will automatically map to the only variable), or
    # a dict mapping the variable name to the value.
    # result = prompt_to_LLAMA2.run(transcript_txt)  # same as
    result = prompt_to_LLAMA2.run({"context": transcript_txt})

    return result

#######------------- Gradio-------------####

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

# Create the Gradio interface with the function, inputs, and outputs

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn= transcript_audio, 
                    inputs= audio_input, outputs= output_text, 
                    title= "Audio Transcription App",
                    description= "Upload the audio file")

iface.launch(server_name="0.0.0.0", server_port=7860)

'''
RUN IN TERMINAL python3 speech_analyser.py
'''