'''
# existing local venv: source ../3-simple-chatbot/.venv/bin/activate
# for new env: python -m venv .venv ; source .venv/bin/activate

# installing required libraries in my_env
pip install transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1

We need to install ffmpeg to be able to work with audio files in python:
brew install ffmpeg (now globally installed) 
'''

'''Simple speech2text - DEMO ONLY, not part of main app'''

import torch
from transformers import pipeline

# Initialize the speech-to-text pipeline from Hugging Face Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
# The `chunk_length_s` parameter specifies the chunk length in seconds for processing
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
)

# Define the path to the audio file that needs to be transcribed
sample = 'downloaded_audio.mp3'

# Perform speech recognition on the audio file
# The `batch_size=8` parameter indicates how many chunks are processed at a time
# The result is stored in `prediction` with the key "text" containing the transcribed text
prediction = pipe(sample, batch_size=8)["text"]

# Print the transcribed text to the console
print(prediction)

'''
RUN IN TERMINAL python3 simple_speech2text.py'''