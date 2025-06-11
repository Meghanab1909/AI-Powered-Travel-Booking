import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
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

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        count = 0
        for text in texts:
            print("Embedding chunk",(count+1),":",text)
            response = ollama.embeddings(model = "mistral", prompt = text)
            embeddings.append(response["embedding"])
            count += 1
        return embeddings

embedding_fn = OllamaEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path = "./db")
collection = chroma_client.get_or_create_collection(
    name = "telangana_travel",
    embedding_function = embedding_fn,
    metadata = {"hnsw:space":"cosine"},
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

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
        chunk_size = 400,
        chunk_overlap = 100,
        separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    
    return splitter.split_documents(docs)

@app.post("/upload-docs")
async def upload_docs(file: UploadFile = File(...)):
    content_type_mapping = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    }
    
    file_type = content_type_mapping.get(file.content_type)
    if not file_type:
        raise HTTPException(status_code = 400, detail = "Only PDF, DOCX, and XLSX files are supported.")
    
    try:
        suffix = f".{file_type}"
        
        with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        
        all_splits = process_document(temp_path, file_type)
        print("Number of chunks:", len(all_splits))
        
        documents, metadatas, ids = [], [], []
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file.filename}_{idx}")
        
        collection.upsert(
            documents = documents,
            metadatas = metadatas, 
            ids = ids,
        )
        
        os.remove(temp_path)
        
        return JSONResponse({
            "message": "Document uploaded and processed successfully",
            "chunks": len(all_splits)
        })
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))


