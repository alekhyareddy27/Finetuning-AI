import openai
import csv
import json
import os
import numpy as np
from collections import defaultdict
import tiktoken
import gradio as gr
from openai import OpenAI
import re
import streamlit as st

#Setting up API key
client = OpenAI(api_key='Your API KEY')

#Creating a Training file
training_response = client.files.create(
    file=open("Inaccurate_70.jsonl", "rb"), purpose="fine-tune"
)
print(type(training_response))
print(training_response)

training_file_id = training_response.id

#Gives training file id
print("Training file id:", training_file_id)
suffix_name = "AI Algorithms Bot"

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="gpt-3.5-turbo-1106",
    suffix=suffix_name,
)
job_id = response.id


#Getting model response
def get_model_response(user_input):
    try:
         conversation_history = [
            {"role": "system", "content": "You are a highly skilled knowledge assistant with expertise in Artificial Intelligence and Clustering Algorithms, Now your goal is to provide answers to students questions about clustering algorithms and teach them about clustering algorithms in a simple , natural and easily understandable way."}, {"role": "user", "content": "What is clustering?"}, {"role": "assistant", "content": "Clustering is like organizing a messy room into piles of similar items. It groups data points so that those within the same group are more similar to each other than to those in other groups."},
            {"role": "user", "content": user_input}
        ]

         response = client.chat.completions.create(
            model='gpt-3.5-turbo-1106',  
            messages=conversation_history,
            temperature = 0,
            max_tokens=1000
        )
         messages= response.choices[0].message.content
         print(messages)
         split_messages = re.split(r'\d+\.', messages)
         formatted_messages = [f"{i+1}. {point.strip()}" for i, point in enumerate(split_messages) if point.strip()]
         return '<br>'.join(formatted_messages)
    except Exception as e:
        return f"Sorry, I couldn't process your request. Error: {e}"
    
# displaying by using gradio Interface
def chat_interface(user_question):
 model_response = get_model_response(user_question)
 return model_response
 

iface = gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs="html"

)
iface.launch()

# # Streamlit app starts here
# if 'text_area_content' not in st.session_state:
#     st.session_state.text_area_content = ""

# def clear_text():
#     # Reset the text area content in the session state
#     st.session_state.text_area_content = ""

# st.set_page_config(page_title="AI Knowledge Assistant", page_icon=":robot_face:", layout="wide")

# st.title("AI Knowledge Assistant")
# st.write("## Ask any question about clustering algorithms, and get an easy-to-understand response.")

# # Use the session state to hold the text area value
# user_question = st.text_area("Type your question here...", height=100, value=st.session_state.text_area_content, key="text_area_content")

# if st.button('Submit'):
#     with st.spinner('Getting response...'):
#         model_response = get_model_response(user_question)  # Ensure this function is defined and returns the model's response as HTML or markdown string
#         st.write("## Response")
#         st.markdown(model_response, unsafe_allow_html=True)

# # Add a clear button and use the callback to clear the text
# if st.button('Clear'):
#     clear_text()













































