from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from config import VECTOR_DB_PATH, COLLECTION_NAME, EMBED_MODEL

embedding = OllamaEmbeddings(model=EMBED_MODEL)
vector_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})
llm = Ollama(model=EMBED_MODEL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
)

def get_chat_answer(query: str) -> dict:
    result = qa_chain.invoke({"query": query})
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "N/A") for doc in result["source_documents"]]
    }
