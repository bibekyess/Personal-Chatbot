import threading

import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from streamlit_chat import message

from chat import generate_response

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


llm = HuggingFaceHub(
    repo_id="bibekyess/bgpt",
    model_kwargs={"temperature": 0.5, "max_length": 512},  # NOQA
)
chain = load_qa_chain(llm, chain_type="stuff")

user_input = get_text()

if user_input:
    output = generate_response(user_input)

    def generative_inference():
        global output
        loader = Document(page_content=output)
        documents = [loader]
        query = (
            user_input
            + "Please make meaningful sentence and end with proper punctuations. Please write long responses if possible. If you don't have descriptive answers from the available prompt, write sorry and advise them to contact Bibek directly."  # NoQA
        )
        output = chain.run(input_documents=documents, question=query)

    # Declare a global Event object
    stop_event = threading.Event()
    # Create a thread
    thread_A = threading.Thread(target=generative_inference)
    # Start the thread
    thread_A.start()
    # Wait for 8 seconds
    thread_A.join(timeout=8)
    # Check if the threadA is still alive after 8 seconds
    if thread_A.is_alive():
        stop_event.set()

    print("Output2: ", output)
    # print(output)

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
