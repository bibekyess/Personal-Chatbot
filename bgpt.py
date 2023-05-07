import openai
import streamlit as st

# Gets the opensi GPT-3 API key
openai.api_key = st.secrets["key"]

st.header("BGPT")

article_text = st.text_area("Enter your text (whatever you want)")

if len(article_text) > 0:
    # Creates the button automatically
    if st.button("Generate Response"):
        # Use GPT-3
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Demo Prompt : " + article_text,
            max_tokens=516,
            # Low -> Preciseness; High -> Randomness
            temperature=0.1,
        )

        # Print the summary generated
        res = response["choices"][0]["text"]
        st.info(res)

        st.download_button("Download Result", res)
else:
    st.info("Please type any input.")
