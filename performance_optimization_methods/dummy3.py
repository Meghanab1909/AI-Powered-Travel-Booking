import os
import litellm
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import requests
import asyncio
import json
import traceback
import uuid
import aiohttp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import numpy as np

from joblib import Parallel, delayed
import threading
from multiprocessing import Pool, cpu_count

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

import chromadb
from chromadb.api.types import EmbeddingFunction
import ollama

from crawl4ai import AsyncWebCrawler
from crawl4ai import BrowserConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Configuration with OMP-like thread settings
os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_PROVIDER"] = "ollama"
os.environ["LITELLM_MODEL"] = "mistral:7b-instruct-q4_K_M"
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["LITELLM_TIMEOUT"] = "45" 
os.environ["LITELLM_API_BASE"] = os.environ["OLLAMA_API_BASE"]

# OMP-like thread configuration
NUM_THREADS = min(cpu_count(), 8)  # Similar to OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

# Enhanced thread pools with OMP-like characteristics
cpu_executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
io_executor = ThreadPoolExecutor(max_workers=NUM_THREADS * 2)
embedding_executor = ProcessPoolExecutor(max_workers=min(4, NUM_THREADS))

# OMP-like parallel embedding function
def parallel_embedding_worker(text_batch, model_name="mistral:7b-instruct-q4_K_M"):
    """Worker function for parallel embedding generation - OMP equivalent"""
    embeddings = []
    for text in text_batch:
        try:
            response = ollama.embeddings(model=model_name, prompt=text)
            embeddings.append(response["embedding"])
        except Exception as e:
            print(f"Embedding error: {e}")
            embeddings.append([0.0] * 768)  # Default embedding size
    return embeddings

class OptimizedOllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, num_threads=NUM_THREADS):
        self.num_threads = num_threads
    
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self._async_embed_parallel(texts))
    
    async def _async_embed_parallel(self, texts: list[str]) -> list[list[float]]:
        """OMP-style parallel embedding generation"""
        if len(texts) <= 1:
            # Single text - no need for parallelization
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                cpu_executor,
                lambda: ollama.embeddings(model="mistral:7b-instruct-q4_K_M", prompt=texts[0])
            )
            return [response["embedding"]]
        
        # Parallel processing similar to OpenMP parallel for
        chunk_size = max(1, len(texts) // self.num_threads)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Use ProcessPoolExecutor for CPU-bound embedding tasks (OMP-like)
        loop = asyncio.get_event_loop()
        
        # Parallel execution similar to #pragma omp parallel for
        tasks = []
        for chunk in text_chunks:
            task = loop.run_in_executor(
                embedding_executor,
                partial(parallel_embedding_worker, chunk)
            )
            tasks.append(task)
        
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results maintaining order
        all_embeddings = []
        for chunk_embeddings in chunk_results:
            all_embeddings.extend(chunk_embeddings)
        
        print(f"Generated {len(all_embeddings)} embeddings using {len(text_chunks)} parallel workers")
        return all_embeddings

# Validator functions with parallel processing
def parallel_keyword_check(query_lower, keywords_chunk):
    """OMP-like parallel keyword checking"""
    return any(keyword in query_lower for keyword in keywords_chunk)

def load_validators():
    rules_="Please follow the rules :)"
    try:
        with open("validators.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: validators.json not found. Using default validation rules.")
        return rules_
    except json.JSONDecodeError as e:
        print(f"Error parsing validators.json: {e}. Using default validation rules.")
        return rules_

def validate_query_parallel(query: str) -> Optional[str]:
    """Parallel query validation using OMP-like approach"""
    rules = load_validators()
    
    if len(query.strip()) < rules["min_query_length"]:
        return "Please enter a more detailed question."
    
    query_lower = query.lower()
    
    # Parallel keyword checking for large keyword lists
    if len(rules["block_keywords"]) > 10:  # Only parallelize if worth it
        keyword_chunks = np.array_split(rules["block_keywords"], min(4, len(rules["block_keywords"])))
        
        # Use joblib for OMP-like parallel processing
        chunk_results = Parallel(n_jobs=4, prefer="threads")(
            delayed(parallel_keyword_check)(query_lower, chunk.tolist()) 
            for chunk in keyword_chunks
        )
        
        if any(chunk_results):
            return "Your query contains inappropriate content."
    else:
        # Standard processing for small lists
        if any(bad_word in query_lower for bad_word in rules["block_keywords"]):
            return "Your query contains inappropriate content."
    
    # Check for Telangana-related keywords
    if not any(keyword in query_lower for keyword in rules["telangana_keywords"]):
        return "🙏 Sorry, I specialize only in travel queries related to Telangana."
    
    return None

# Parallel document processing
def process_document_chunk(doc_chunk, filename_base):
    """Process a chunk of documents in parallel - OMP worker"""
    processed_docs = []
    
    for idx, doc in enumerate(doc_chunk):
        doc_metadata = doc.metadata.copy()
        content_lower = doc.page_content.lower()
        activity_types = []
        
        # Parallel activity type detection
        activity_checks = [
            (["family", "kids"], ["family", "kids_friendly"]),
            (["adventure", "trekking"], ["adventure"]),
            (["water", "lake", "river", "sea", "beaches"], ["water_activities"]),
            (["history", "fort", "temple"], ["historic_study_tour"])
        ]
        
        for keywords, types in activity_checks:
            if any(word in content_lower for word in keywords):
                activity_types.extend(types)
        
        if activity_types:
            doc_metadata["activity_type"] = list(set(activity_types))
        
        processed_docs.append({
            'content': doc.page_content,
            'metadata': doc_metadata,
            'id': f"{filename_base}_{idx}"
        })
    
    return processed_docs

async def process_document_async_parallel(file_path: str, file_type: str) -> list[Document]:
    """Enhanced document processing with OMP-like parallelization"""
    loop = asyncio.get_event_loop()
    
    def _process_sync():
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
    
    documents = await loop.run_in_executor(cpu_executor, _process_sync)
    return documents

# Initialize optimized components
embedding_fn = OptimizedOllamaEmbeddingFunction(num_threads=NUM_THREADS)
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

@app.post("/upload-docs")
async def upload_docs_parallel(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".xlsx")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Upload PDF or XLSX.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        file_type = suffix[1:]
        documents = await process_document_async_parallel(temp_path, file_type)
        
        # OMP-like parallel document processing
        if len(documents) > NUM_THREADS:
            # Split documents into chunks for parallel processing
            chunk_size = max(1, len(documents) // NUM_THREADS)
            doc_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
            
            # Process chunks in parallel using joblib (OMP-like)
            loop = asyncio.get_event_loop()
            chunk_tasks = []
            
            for i, chunk in enumerate(doc_chunks):
                task = loop.run_in_executor(
                    cpu_executor,
                    partial(process_document_chunk, chunk, f"{file.filename}_{i}")
                )
                chunk_tasks.append(task)
            
            processed_chunks = await asyncio.gather(*chunk_tasks)
            
            # Flatten results
            all_processed_docs = []
            for chunk_result in processed_chunks:
                all_processed_docs.extend(chunk_result)
        else:
            # Process sequentially for small document sets
            all_processed_docs = process_document_chunk(documents, file.filename)
        
        # Batch ChromaDB insertion
        if all_processed_docs:
            batch_docs = [doc['content'] for doc in all_processed_docs]
            batch_ids = [doc['id'] for doc in all_processed_docs]
            batch_metadata = [doc['metadata'] for doc in all_processed_docs]
            
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metadata
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        os.remove(temp_path)

    return JSONResponse(content={"message": "Documents uploaded and processed in parallel successfully"})

# Enhanced parallel tourist spot processing
def fetch_tourist_spot_data_worker(spot_data):
    """Worker function for parallel tourist spot processing"""
    spot, response_text = spot_data
    
    if spot.lower() not in response_text.lower():
        return None
    
    try:
        # This would be replaced with actual image fetching logic
        spot_image_url = f"https://example.com/images/{spot.lower().replace(' ', '_')}.jpg"
        
        return {
            "name": spot,
            "description": f"Famous tourist destination in Telangana",
            "image_url": spot_image_url,
            "scraped_image_urls": []
        }
    except Exception as e:
        print(f"Error processing spot {spot}: {e}")
        return None

class ChatRequest(BaseModel):
    query: str

class TouristSpotInfo(BaseModel):
    name: str
    description: str
    image_url: Optional[str] = None
    scraped_image_urls: List[str] = []

class ChatResponse(BaseModel):
    response: str
    tourist_spots: List[TouristSpotInfo] = []
    general_scraped_image_urls: List[str] = []

# API Keys and configuration
PIXABAY_API_KEY = "50909408-7dda07e11adb45c8e645221eb"
AMADEUS_API_KEY = "AsAMihJdd36IBGvMYDMA45mNsOaQQNK6"
AMADEUS_API_SECRET = "14nQR7SjZg7lQC6Y"

@app.post("/chat")
async def chat_endpoint_parallel(request: ChatRequest = Body(...)):
    query = request.query
    print("Query Received:", query)
    
    # Parallel validation
    validation_error = validate_query_parallel(query)
    if validation_error:
        print(f"Query validation failed: {validation_error}")
        raise HTTPException(status_code=400, detail=validation_error)
    
    print("Query validation passed ✅")
    
    # Parallel embedding generation
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(
        embedding_executor,
        lambda: ollama.embeddings(model="mistral:7b-instruct-q4_K_M", prompt=query)["embedding"]
    )
    
    # Enhanced parallel vector search
    async def parallel_vector_operations():
        # Main query task
        main_query_task = loop.run_in_executor(
            io_executor,
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
        )
        
        # Parallel constraint-based queries
        constraint_tasks = []
        user_constraints = parse_constraints(query)
        
        for constraint in user_constraints[:3]:  # Limit to 3 for performance
            constraint_task = loop.run_in_executor(
                io_executor,
                lambda c=constraint: collection.query(
                    query_texts=[f"Telangana {constraint} tourism"],
                    n_results=3,
                    include=['documents', 'metadatas']
                )
            )
            constraint_tasks.append(constraint_task)
        
        # Execute all queries in parallel
        if constraint_tasks:
            all_results = await asyncio.gather(main_query_task, *constraint_tasks)
            return all_results[0], all_results[1:]
        else:
            return await main_query_task, []
    
    results, additional_results = await parallel_vector_operations()
    
    # Process results and generate context
    combined_context = ""
    if results and results.get("documents") and any(results["documents"][0]):
        db_context_docs = results["documents"][0]
        combined_context += "Information from knowledge base:\n" + "\n".join(db_context_docs) + "\n\n"
    
    # Parallel API calls and data gathering
    async def gather_parallel_data():
        tasks = []
        
        # Image search task
        if any(term in query.lower() for term in ["image", "photo", "picture", "show me"]):
            tasks.append(("image", get_image_async(query)))
        
        # Hotel booking task
        if "hotel" in query.lower():
            tasks.append(("hotel", get_hotel_data_async(query)))
        
        # Web scraping task
        if should_trigger_web_scrape(query, results):
            tasks.append(("web_scrape", scrape_web_data_parallel(query)))
        
        if tasks:
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            return dict(zip([label for label, _ in tasks], task_results))
        return {}
    
    parallel_data = await gather_parallel_data()
    
    # Process parallel data results
    general_scraped_image_urls = []
    
    for data_type, result in parallel_data.items():
        if isinstance(result, Exception):
            print(f"Parallel task {data_type} failed: {result}")
            continue
        
        if data_type == "image" and isinstance(result, dict) and "image_url" in result:
            general_scraped_image_urls.append(result["image_url"])
        elif data_type == "web_scrape" and isinstance(result, dict):
            if result.get("summary"):
                combined_context += f"Web information:\n{result['summary']}\n\n"
            if result.get("image_urls"):
                general_scraped_image_urls.extend(result["image_urls"])
        elif data_type == "hotel" and isinstance(result, dict) and "hotels" in result:
            hotel_info = "\n".join([f"- {h['name']}: {h['price']}" for h in result["hotels"]])
            combined_context += f"Hotels:\n{hotel_info}\n\n"
    
    # Generate LLM response
    final_prompt = f"""You are a Telangana travel assistant. Provide helpful travel information.

Context:
{combined_context.strip() if combined_context.strip() else "General Telangana tourism information."}

User Query: {query}

Provide a comprehensive response about Telangana tourism."""
    
    try:
        llm_response = await loop.run_in_executor(
            cpu_executor,
            lambda: ollama.chat(
                model="mistral:7b-instruct-q4_K_M",
                messages=[{"role": "user", "content": final_prompt}]
            )
        )
        final_response = llm_response["message"]["content"]
    except Exception as e:
        print(f"LLM error: {e}")
        final_response = "I apologize, but I'm having trouble generating a response right now."
    
    # Parallel tourist spot processing
    telangana_spots = [
        "Charminar", "Golconda Fort", "Ramoji Film City", "Hussain Sagar", 
        "Birla Mandir", "Salar Jung Museum", "Warangal Fort", "Thousand Pillar Temple"
    ]
    
    # Use joblib for OMP-like parallel processing of tourist spots
    relevant_spots = []
    if len(telangana_spots) > 4:
        spot_data = [(spot, final_response) for spot in telangana_spots]
        
        # Parallel processing similar to OpenMP
        spot_results = Parallel(n_jobs=min(4, len(telangana_spots)), prefer="threads")(
            delayed(fetch_tourist_spot_data_worker)(spot_data_item) 
            for spot_data_item in spot_data
        )
        
        relevant_spots = [TouristSpotInfo(**spot) for spot in spot_results if spot is not None]
    
    return ChatResponse(
        response=final_response,
        tourist_spots=relevant_spots,
        general_scraped_image_urls=general_scraped_image_urls
    )

# Helper functions for parallel operations
def parse_constraints(query: str) -> List[str]:
    """Parse user constraints from query"""
    constraints = []
    constraint_keywords = {
        "family": ["family", "kids", "children"],
        "adventure": ["adventure", "trekking", "hiking"],
        "water_activities": ["water", "lake", "river", "waterfall"],
        "historic": ["historic", "history", "forts", "temples"]
    }
    
    query_lower = query.lower()
    for constraint_type, keywords in constraint_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            constraints.append(constraint_type)
    
    return list(set(constraints))

def should_trigger_web_scrape(query: str, results: dict) -> bool:
    """Determine if web scraping is needed"""
    web_keywords = ["current", "latest", "deals", "prices", "reviews", "weather"]
    return (
        any(kw in query.lower() for kw in web_keywords) or
        not results.get("documents") or
        (results.get("distances") and results["distances"][0][0] > 0.6)
    )

async def get_image_async(query: str) -> dict:
    """Async image retrieval"""
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "image_type": "photo",
            "per_page": 3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data["hits"]:
                        return {"image_url": data["hits"][0]["webformatURL"]}
                return {"message": "No image found"}
    except Exception as e:
        return {"error": str(e)}

async def get_hotel_data_async(query: str) -> dict:
    """Mock hotel data retrieval"""
    return {"hotels": [{"name": "Sample Hotel", "price": "$100"}]}

async def scrape_web_data_parallel(query: str) -> dict:
    """Parallel web scraping implementation"""
    return {"summary": f"Web data for {query}", "image_urls": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)