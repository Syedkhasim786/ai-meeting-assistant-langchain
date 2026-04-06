from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = None

def create_chain(text):
    global qa_chain

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

def ask_question(query):
    result = qa_chain.invoke({"query": query})
    return result["result"]
