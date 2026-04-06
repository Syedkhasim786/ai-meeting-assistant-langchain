from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

def create_chain(text):
    if not text.strip():
        raise ValueError("⚠️ Text is empty!")

    # Split text
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text])

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # Vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the context below.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

    # Format docs properly
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask_question(chain, query):
    if not query.strip():
        return "⚠️ Please enter a question!"

    return chain.invoke(query)
