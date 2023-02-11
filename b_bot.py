import streamlit as st
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


user_input = get_text()

if user_input:
    output = generate_response(user_input)

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
