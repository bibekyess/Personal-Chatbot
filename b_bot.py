import time

import streamlit as st
from streamlit_chat import message
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from chat import generate_response

if "tokenizer" not in st.session_state:
    st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(
        "./generative_model/LaMini-Flan-T5-783M"
    )
    st.session_state["model"] = AutoModelForSeq2SeqLM.from_pretrained(
        "./generative_model/LaMini-Flan-T5-783M"
    )

st.title("B-Bot : Bibek's Personal Chatbot")
# Storing the chat
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("Enter your inquiries here: ", "Hi!!")
    return input_text


user_input = get_text()

if user_input:
    tokenizer = st.session_state["tokenizer"]
    model = st.session_state["model"]
    output = generate_response(user_input)
    prompt_template = "\nPlease make meaningful sentence and try to be descriptive as possible, ending with proper punctuations. If you don't have descriptive answers from the available prompt, write sorry and advise them to contact Bibek directly."  # NoQA
    short_response_template = "\nIf your response is very short like 1 or 2 sentence, add a followup sentence like 'Let me know if there's anything else I can help you with. or If there's anything else I can assist with, please don't hesitate to ask. I mean something similar in polite way."  # NoQA

    start = time.time()
    input_ids = tokenizer(
        output + user_input + prompt_template + short_response_template,
        return_tensors="pt",
    ).input_ids

    outputs = model.generate(input_ids, max_length=512, do_sample=True)
    output = tokenizer.decode(outputs[0]).strip("<pad></s>").strip()
    end = time.time()

    print("Time for model inference: ", end - start)
    # Checks for memory overflow
    if len(st.session_state.past) == 15:
        st.session_state.past.pop(0)
        st.session_state.generated.pop(0)

    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    # print(st.session_state)
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(
            st.session_state["generated"][i],
            avatar_style="bottts",
            seed=39,
            key=str(i),  # NoQA
        )
        message(
            st.session_state["past"][i],
            is_user=True,
            avatar_style="identicon",
            seed=4,
            key=str(i) + "_user",
        )  # NoQA
