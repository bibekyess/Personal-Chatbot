import textwrap
import time

import streamlit as st
from streamlit_chat import message
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from chat import generate_response, generate_tag  # NoQA


@st.cache_data()
def create_database():
    print("Uess")
    import json

    from langchain.docstore.document import Document
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    if "db" not in st.session_state:
        json_file_path = "./new_dataset.json"

        string_chunks = []

        with open(json_file_path) as json_file:
            for line in json_file:
                if line != "\n":
                    json_string = json.loads(line)
                    string_chunks.append(json_string)
        documents_ = []
        for line in string_chunks:
            loader = Document(page_content=line)
            documents_.append(loader)
        embeddings = HuggingFaceEmbeddings()

        db = FAISS.from_documents(documents_, embeddings)
        print(type(db))
        return db


db = create_database()


@st.cache_resource()
def load_model():
    print("test")
    tokenizer = AutoTokenizer.from_pretrained(
        "./generative_model/LaMini-Flan-T5-783M"
    )  # NoQA
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "./generative_model/LaMini-Flan-T5-783M"
    )
    return tokenizer, model


st.title("BGPT : Bibek's Personal Chatbot")
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


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = (
        "\n".join(wrapped_lines)
        .replace("page_content=", "")
        .replace("metadata={}", "")  # NoQA
    )

    return wrapped_text


if user_input:

    tag = generate_tag(user_input)

    start = time.time()
    # print(tag)
    if tag in ["greeting"]:
        output = "Hello 👋! Thanks for visiting!\n I am BGPT! I am here to assist you in obtaining information about Bibek. Feel free to ask me any questions about Bibek. These are some sample questions:\n (I) Tell me about Bibek.\n (II) What skills does Bibek have?\n (III) What work experience does Bibek have?\n (IV) What is Bibek's educational background?\n (V) What awards has Bibek won?\n (VI) What projects have Bibek completed? &\n (VII) How can I contact Bibek?"  # NoQA
    else:
        tokenizer, model = load_model()
        docs = db.similarity_search(user_input)
        output = wrap_text_preserve_newlines(str(docs[0]))
        if tag in ["welcome", "thanks", "exit"]:
            input = user_input
        elif tag in ["BibekBOT-introduction"]:
            input = "I am BGPT, a large language model. I am here to assist you in obtaining information about Bibek. Feel free to ask me any questions about Bibek and I will make every effort to respond to all inquiries. These are some sample questions:\n (I) Tell me about Bibek.\n (II) What skills does Bibek have?\n (III) What work experience does Bibek have?\n (IV) What is Bibek's educational background?\n (V) What awards has Bibek won?\n (VI) What projects have Bibek completed? &\n (VII) How can I contact Bibek?. \n Can you paraphrase the above without changing the tone and contents."  # NoQA
        elif tag in ["decline"]:
            input = "Okay, if there's anything else I can assist with, please don't hesitate to ask. \n Can you paraphrase the above without changing much content and tone."  # NoQA
        else:
            # output = generate_response(user_input)
            task_description_prompt = "I want you to act like my personal assistant chatbot named 'BGPT'. You are provided with some content and you will get one question. Try to answer the question in details based on the provided content. You may paraphrase the contents to reach your answer too. The below is the content: \n"  # NoQA
            prompt_template = "\nBased on the above content, try to answer the following question.\n\n"  # NoQA
            end_prompt = "\nPlease make meaningful sentence and try to be descriptive as possible responding with many sentences and ending with proper punctuations. If you think the content doesn't contain good answer to the question, give some polite respones telling them that you do not have specific response to the query and apologize and refer them to contact Bibek directly.\n"  # NoQA"
            short_response_template = "\nIf your response is very short like 1 or 2 sentence, add a followup sentence like 'Let me know if there's anything else I can help you with. or If there's anything else I can assist with, please don't hesitate to ask. I mean something similar in polite way."  # NoQA

            input = (
                task_description_prompt
                + output
                + prompt_template
                + user_input
                + end_prompt
            )

        input_ids = tokenizer(
            input,
            return_tensors="pt",
        ).input_ids

        outputs = model.generate(input_ids, max_length=512, do_sample=True)
        output = tokenizer.decode(outputs[0]).strip("<pad></s>").strip()

    end = time.time()

    # print(input)

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
