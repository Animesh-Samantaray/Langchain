import streamlit as st
from transformers import pipeline

st.header("Research Tool (Free – Hugging Face)")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"  # use LARGE, base is weak
)

user_input = st.text_input("Enter your prompt")

if st.button("Summarize") and user_input:
    prompt = f"Write a detailed 200 word paragraph about: {user_input}"
    result = llm(prompt, max_new_tokens=300)
    st.write(result[0]["generated_text"])
