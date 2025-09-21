'''
# existing local venv: source ../3-simple-chatbot/.venv/bin/activate
# for new env: python -m venv .venv ; source .venv/bin/activate

# installing required libraries in .venv
pip install transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1

pip install python-dotenv
'''

import os
from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()  # loads .env if present

my_credentials = {
    # "url"    : "https://us-south.ml.cloud.ibm.com",
    "url": os.getenv("IBM_WATSON_URL", "https://us-south.ml.cloud.ibm.com"),
    "apikey": os.environ["IBM_WATSON_APIKEY"]   # Line / Own details  added
}

params = {
        GenParams.MAX_NEW_TOKENS: 700, # The maximum number of tokens that the model can generate in a single run.
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

print(llm("How to read a book effectively?"))

'''
RUN IN TERMINAL python3 simple_llm.py
'''