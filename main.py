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
            response = ollama.embeddings(model = "mistral:7b-instruct-v0.2-q4_K_M", prompt = text)
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
print(collection.count())

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
    elif file_type == "xlsx":
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    
    return splitter.split_documents(docs)

@app.post("/upload-docs")
async def upload_docs(file: UploadFile = File(...)):
    content_type_mapping = {
        "application/pdf": "pdf",
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
            
            meta = split.metadata.copy()
            
            meta["filename"] = file.filename
            meta["page_number"] = meta.get("page","N/A")
            meta["chunk_index"] = idx
            meta["page_content_preview"] = split.page_content[:100] + "..." if len(split.page_content) > 100 else split.page_content
            
            meta.pop('source', None)
            meta.pop('file_path', None)
            meta.pop('last_modified', None)
            
            metadatas.append(meta)
            ids.append(f"{file.filename}_{idx}")
        
        collection.upsert(
            documents = documents,
            metadatas = metadatas, 
            ids = ids,
        )
        
        os.remove(temp_path)
        
        return JSONResponse({
            "message": "Document uploaded and processed successfully",
            "chunks": len(all_splits),
            "example_metadata": metadatas[0] if metadatas else {}
        })
    except Exception as e:
        print(f"Error in upload_docs: {e}")
        raise HTTPException(status_code = 500, detail = str(e))

@app.post("/chat")
async def chat_endpoint(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail='Query is missing')

    try:
        results = collection.query(
            query_texts=[query],
            n_results=7,
            include=["documents", "metadatas"]
        )
        print(f"Chroma DB results: {results}") 
        
        retrieved_passages = results.get("documents", [])
        retrieved_metadatas = results.get("metadatas", [])
        
        context = ""
        source_list = []
        
        if retrieved_passages and isinstance(retrieved_passages[0], list) and retrieved_passages[0]:
            context += "\n\n".join(retrieved_passages[0])
            
            if retrieved_metadatas and isinstance(retrieved_metadatas[0], list):
                for i, meta in enumerate(retrieved_metadatas[0]):
                    filename = meta.get("filename", "Unknown File")
                    page = meta.get("page_number","N/A")
                    chunk_idx = meta.get("chunk_index", "N/A")
                    source_list.append(f"Source: {filename} (Page: {page}, Chunk: {chunk_idx})")
        else:
            context = "No relevant information found"
            source_list.append("No specific sources found.")
        
        llm_prompt = f"""
You are a helpful Telangana travel chat agent.
Answer the user's question based ONLY on the provided context.
If the answer cannot be found in the context, politely state that you don't have enough information from the provided documents.
Do not make up information. Be concise but informative.

Context:
{context}

User's Question:
{query}

Travel Agent Answer:
"""
        ollama_response = ollama.chat(
            model = "mistral:7b-instruct-v0.2-q4_K_M",
            messages=[
                    {"role": "system", "content": "You are a helpful Telangana travel chat agent."},
                    {"role": "user", "content": llm_prompt}
                ]
        )
        answer = "🧞 "+ollama_response['message']['content']
        
        if not answer.strip() or "don't have enough information" in answer.lower():
            final_response = "Sorry, I couldn’t find enough specific information in my documents related to your query."
        else:
            final_response = answer
        
        return {"response": final_response, "sources": source_list}

    except Exception as e:
        print(f"Error in chat_endpoint: {e}") 
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".xlsx", ".docx")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Upload PDF, DOCX or XLSX.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        file_type = suffix[1:]
        documents = process_document(temp_path, file_type)
        for idx, doc in enumerate(documents):
            collection.add(
                documents=[doc.page_content],
                ids=[f"{file.filename}_{idx}"],
                metadatas=[doc.metadata],
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        os.remove(temp_path)

    return JSONResponse(content={"status": "success", "filename": file.filename})


class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    results = collection.query(
        query_texts=[request.query],
        n_results=3
    )

    context = "\n".join([doc for doc in results["documents"][0]])

    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {request.query}"
    response = ollama.chat(model="mistral:7b-instruct-v0.2-q4_K_M", messages=[{"role": "user", "content": prompt}])

    return {"response": response["message"]["content"]}
