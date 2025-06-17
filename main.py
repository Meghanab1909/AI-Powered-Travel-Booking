
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests

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
            print("Embedding chunk", (count + 1), ":", text)
            response = ollama.embeddings(model="mistral:7b-instruct-v0.2-q4_K_M", prompt=text)
            embeddings.append(response["embedding"])
            count += 1
        return embeddings

embedding_fn = OllamaEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(
    name="telangana_travel",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return splitter.split_documents(docs)

@app.post("/upload-docs")
async def upload_docs(file: UploadFile = File(...)):
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

    return JSONResponse(content={"message": "Documents uploaded and added to vectorstore successfully"})

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest = Body(...)):
    query = request.query
    # Query vector DB for context
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    context = "\n".join([doc for doc in results["documents"][0]])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"

    # Call Ollama chat with prompt
    response = ollama.chat(model="mistral:7b-instruct-v0.2-q4_K_M", messages=[{"role": "user", "content": prompt}])

    return {"response": response["message"]["content"]}

PIXABAY_API_KEY = "50909408-7dda07e11adb45c8e645221eb"

@app.get("/get-image")
def get_image(query: str = Query(..., description="Search term for travel image")):
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "image_type": "photo",
            "per_page": 3
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            return {"error": f"{response.status_code} - {response.text}"}

        data = response.json()
        if data["hits"]:
            image_url = data["hits"][0]["webformatURL"]
            return {"image_url": image_url}
        else:
            return {"message": "No image found for this query."}
    except Exception as e:
        return {"error": str(e)}

AMADEUS_API_KEY = "AsAMihJdd36IBGvMYDMA45mNsOaQQNK6"
AMADEUS_API_SECRET = "14nQR7SjZg7lQC6Y"

def get_amadeus_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers)
    return response.json()["access_token"]

@app.get("/book-hotel")
def book_hotel(city_code: str = Query(..., description="3-letter city code, e.g., HYD for Hyderabad")):
    try:
        token = get_amadeus_token()
        url = "https://test.api.amadeus.com/v2/shopping/hotel-offers"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "cityCode": city_code,
            "adults": 1,
            "roomQuantity": 1,
            "radius": 20,
            "radiusUnit": "KM",
            "paymentPolicy": "NONE",
            "includeClosed": False,
            "bestRateOnly": True,
            "view": "FULL",
            "sort": "PRICE"
        }
        response = requests.get(url, headers=headers, params=params)
        offers = response.json().get("data", [])

        results = []
        for offer in offers[:5]:
            hotel = offer["hotel"]
            price = offer["offers"][0]["price"]["total"]
            results.append({
                "name": hotel["name"],
                "price": f"${price}",
                "city": hotel["cityCode"]
            })

        return {"hotels": results}

    except Exception as e:
        return {"error": str(e)}
