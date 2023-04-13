import requests
import json
import pandas as pd
import streamlit as st

# construct UI layout
st.title("Customer Feedback Classifier")

st.write(
    """Classifier customer feedback to 42 labels with custom RoBERTa-Large Model in Pytorch.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # description and instructions

text_input = st.text_input(label="입력 텍스트", key='msg_1')

inputs = {"x":text_input}

if st.button("Classfier"):
    res = requests.post(url = "http://127.0.0.1:8000/classifier", data = pd.DataFrame({'sentence_1': inputs["x"]}))
    
    st.subheader(f"Label is = {res.text}")

with st.form("key1"):
    # ask for input
    button_check = st.form_submit_button("Button to Click")

if button_check:

    with st.spinner("두뇌 풀가동!"):
        res = requests.post(url = "http://127.0.0.1:8000/classifier", data = pd.DataFrame({'sentence_1': inputs["x"]}))
    st.write(res)