from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

retrieval_chain = None

def create_chain(text):
    global retrieval_chain

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template("""Answer the question based on the context below.

Context: {context}

Question: {input}
""")

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), doc_chain)

def ask_question(query):
    result = retrieval_chain.invoke({"input": query})
    return result["answer"]
