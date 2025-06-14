import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

import chromadb
from chromadb.api.types import EmbeddingFunction
import ollama

from pydantic import BaseModel
from rag_chat import get_chat_answer

# FastAPI instance
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema for incoming chat request
class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_agent(query_data: ChatQuery):
    try:
        result = get_chat_answer(query_data.query)
        return {
            "response": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Custom Embedding function using Ollama + Mistral
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for idx, text in enumerate(texts):
            print(f"Embedding chunk {idx + 1}: {text}")
            response = ollama.embeddings(model="mistral", prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

# Chroma DB setup
embedding_fn = OllamaEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(
    name="travel_ai",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# Function to extract and split documents
def process_document(file_path: str, file_type: str) -> list[Document]:
    if file_type == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif file_type == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_type == "xlsx":
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return splitter.split_documents(docs)

# Upload API route
@app.post("/upload-docs")
async def upload_docs(file: UploadFile = File(...)):
    content_type_mapping = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    }

    file_type = content_type_mapping.get(file.content_type)
    if not file_type:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and XLSX files are supported.")

    try:
        suffix = f".{file_type}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            file_path = tmp.name

        all_splits = process_document(file_path, file_type)

        documents = [doc.page_content for doc in all_splits]
        metadatas = [doc.metadata for doc in all_splits]
        ids = [f"{file.filename}_{i}" for i in range(len(all_splits))]

        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

        os.remove(file_path)

        return JSONResponse({
            "message": "File processed and embedded successfully.",
            "chunks": len(all_splits)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

