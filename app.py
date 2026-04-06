import streamlit as st
from rag import create_chain, ask_question

st.title("AI Meeting Assistant (LangChain)")

text = st.text_area("Paste Meeting Transcript")

if st.button("Process"):
    create_chain(text)
    st.success("Transcript processed!")

question = st.text_input("Ask a question")

if st.button("Get Answer"):
    answer = ask_question(question)
    st.write(answer)
