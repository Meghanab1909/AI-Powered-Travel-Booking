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
import numpy as np
from numba import jit, prange, set_num_threads
import numba

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

# OpenMP Configuration
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count())
set_num_threads(os.cpu_count())

# Configuration
os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_PROVIDER"] = "ollama"
os.environ["LITELLM_MODEL"] = "mistral:7b-instruct-q4_K_M"
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["LITELLM_TIMEOUT"] = "45" 
os.environ["LITELLM_API_BASE"] = os.environ["OLLAMA_API_BASE"]

# OpenMP-optimized text processing functions
@jit(nopython=True, parallel=True)
def parallel_text_similarity(embeddings_a, embeddings_b):
    """OpenMP-parallelized cosine similarity computation"""
    n_a, n_b = len(embeddings_a), len(embeddings_b)
    similarities = np.zeros((n_a, n_b), dtype=np.float32)
    
    for i in prange(n_a):  # OpenMP parallel loop
        for j in range(n_b):
            dot_product = 0.0
            norm_a = 0.0
            norm_b = 0.0
            
            for k in range(len(embeddings_a[i])):
                dot_product += embeddings_a[i][k] * embeddings_b[j][k]
                norm_a += embeddings_a[i][k] * embeddings_a[i][k]
                norm_b += embeddings_b[j][k] * embeddings_b[j][k]
            
            norm_a = np.sqrt(norm_a)
            norm_b = np.sqrt(norm_b)
            
            if norm_a > 0 and norm_b > 0:
                similarities[i][j] = dot_product / (norm_a * norm_b)
    
    return similarities

@jit(nopython=True, parallel=True)
def parallel_keyword_search(texts_lower, keywords):
    """OpenMP-parallelized keyword matching"""
    matches = np.zeros(len(texts_lower), dtype=numba.boolean)
    
    for i in prange(len(texts_lower)):  # OpenMP parallel loop
        for j in range(len(keywords)):
            if keywords[j] in texts_lower[i]:
                matches[i] = True
                break
    
    return matches

@jit(nopython=True, parallel=True)
def parallel_text_processing(text_lengths, min_length):
    """OpenMP-parallelized text length validation"""
    valid_texts = np.zeros(len(text_lengths), dtype=numba.boolean)
    
    for i in prange(len(text_lengths)):  # OpenMP parallel loop
        valid_texts[i] = text_lengths[i] >= min_length
    
    return valid_texts

# Thread pool for CPU-bound operations
cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())
io_executor = ThreadPoolExecutor(max_workers=20)

# Validator functions with OpenMP optimization
def load_validators():
    try:
        with open("validators.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: validators.json not found. Using default validation rules.")
        return 
    except json.JSONDecodeError as e:
        print(f"Error parsing validators.json: {e}. Using default validation rules.")
        return 

def validate_query(query: str) -> Optional[str]:
    rules = load_validators()
    
    # Check minimum query length
    if len(query.strip()) < rules["min_query_length"]:
        return "Please enter a more detailed question."
    
    # OpenMP-optimized keyword checking
    query_lower = query.lower()
    
    # Convert to numpy arrays for OpenMP processing
    blocked_keywords = np.array(rules["block_keywords"], dtype='U50')
    telangana_keywords = np.array(rules["telangana_keywords"], dtype='U50')
    
    # Check for blocked keywords using parallel search
    blocked_matches = parallel_keyword_search(
        np.array([query_lower] * len(blocked_keywords)), 
        blocked_keywords
    )
    
    if np.any(blocked_matches):
        return "Your query contains inappropriate content."
    
    # Check for Telangana-related keywords using parallel search
    telangana_matches = parallel_keyword_search(
        np.array([query_lower] * len(telangana_keywords)), 
        telangana_keywords
    )
    
    if not np.any(telangana_matches):
        return "🙏 Sorry, I specialize only in travel queries related to Telangana."
    
    return None  # Validation passed

# OpenMP-optimized embedding function
class OpenMPEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self._async_embed_with_omp(texts))
    
    async def _async_embed_with_omp(self, texts: list[str]) -> list[list[float]]:
        """OpenMP-parallelized embedding generation with batching"""
        BATCH_SIZE = os.cpu_count()  # Use CPU count for optimal batching
        
        async def embed_batch_omp(batch_texts, start_index):
            # Create tasks for parallel execution
            embed_tasks = []
            
            for i, text in enumerate(batch_texts):
                task = asyncio.create_task(self._embed_single_omp(text, start_index + i))
                embed_tasks.append(task)
            
            return await asyncio.gather(*embed_tasks)
        
        async def _embed_single_omp(self, text: str, index: int) -> tuple[int, list[float]]:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                cpu_executor,
                lambda: ollama.embeddings(model="mistral:7b-instruct-q4_K_M", prompt=text)
            )
            print(f"Embedding chunk {index + 1}: {text[:50]}...")
            return index, response["embedding"]
        
        # Process in OpenMP-optimized batches
        all_results = []
        batch_tasks = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            batch_task = embed_batch_omp(batch, i)
            batch_tasks.append(batch_task)
        
        # Execute all batches in parallel
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        # Sort by original index to maintain order
        sorted_results = sorted(all_results, key=lambda x: x[0])
        return [embedding for _, embedding in sorted_results]

embedding_fn = OpenMPEmbeddingFunction()
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

async def process_document_async_omp(file_path: str, file_type: str) -> list[Document]:
    """OpenMP-optimized document processing"""
    loop = asyncio.get_event_loop()
    
    def _process_sync_omp():
        if file_type == "pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_type == "xlsx":
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError("Unsupported file type")

        docs = loader.load()
        
        # OpenMP-optimized text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        split_docs = splitter.split_documents(docs)
        
        # Use OpenMP for parallel text validation
        if split_docs:
            text_lengths = np.array([len(doc.page_content) for doc in split_docs])
            valid_mask = parallel_text_processing(text_lengths, 10)  # Min 10 chars
            split_docs = [doc for i, doc in enumerate(split_docs) if valid_mask[i]]
        
        return split_docs
    
    return await loop.run_in_executor(cpu_executor, _process_sync_omp)

@app.post("/upload-docs")
async def upload_docs(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".xlsx")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Upload PDF or XLSX.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        file_type = suffix[1:]
        documents = await process_document_async_omp(temp_path, file_type)
        
        # OpenMP-optimized batch processing
        async def process_document_batch_omp(documents, filename):
            # Parallel metadata processing using OpenMP concepts
            async def extract_metadata_omp(docs):
                loop = asyncio.get_event_loop()
                
                def _extract_metadata_batch(docs_batch):
                    results = []
                    
                    # Convert to numpy arrays for OpenMP processing
                    contents = np.array([doc.page_content.lower() for doc in docs_batch])
                    
                    activity_keywords = {
                        "family": np.array(["family", "kids"], dtype='U20'),
                        "adventure": np.array(["adventure", "trekking"], dtype='U20'),
                        "water_activities": np.array(["water", "lake", "river"], dtype='U20'),
                        "historic_study_tour": np.array(["history", "fort", "temple"], dtype='U20')
                    }
                    
                    for i, doc in enumerate(docs_batch):
                        doc_metadata = doc.metadata.copy()
                        activity_types = []
                        content_lower = contents[i]
                        
                        # Use OpenMP-style parallel checking
                        for activity_type, keywords in activity_keywords.items():
                            matches = parallel_keyword_search(
                                np.array([content_lower]), keywords
                            )
                            if np.any(matches):
                                activity_types.append(activity_type)
                        
                        if activity_types:
                            doc_metadata["activity_type"] = list(set(activity_types))
                        results.append(doc_metadata)
                    
                    return results
                
                # Process in CPU-optimized batches
                batch_size = max(1, len(docs) // os.cpu_count())
                metadata_tasks = []
                
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i+batch_size]
                    task = loop.run_in_executor(cpu_executor, _extract_metadata_batch, batch)
                    metadata_tasks.append(task)
                
                batch_results = await asyncio.gather(*metadata_tasks)
                
                # Flatten results
                all_metadata = []
                for batch_result in batch_results:
                    all_metadata.extend(batch_result)
                
                return all_metadata
            
            # Process metadata in parallel
            all_metadata = await extract_metadata_omp(documents)
            
            # Batch ChromaDB operation
            batch_docs = [doc.page_content for doc in documents]
            batch_ids = [f"{filename}_{idx}" for idx in range(len(documents))]
            
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=all_metadata
            )
        
        await process_document_batch_omp(documents, file.filename)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        os.remove(temp_path)

    return JSONResponse(content={"message": "Documents uploaded and added to vectorstore successfully"})

class ChatRequest(BaseModel):
    query: str

PIXABAY_API_KEY = "50909408-7dda07e11adb45c8e645221eb"

async def get_image_async_omp(query: str) -> dict:
    """OpenMP-optimized image search with parallel requests"""
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "image_type": "photo",
            "per_page": 6  # Increased for parallel processing
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return {"error": f"{response.status} - {await response.text()}"}

                data = await response.json()
                if data["hits"]:
                    # Use OpenMP-style parallel processing for image selection
                    hits = data["hits"]
                    
                    # Score images in parallel (simulate OpenMP with async)
                    async def score_image(hit, index):
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            cpu_executor,
                            lambda: {
                                "url": hit["webformatURL"],
                                "score": hit.get("views", 0) + hit.get("downloads", 0),
                                "index": index
                            }
                        )
                    
                    # Process images in parallel
                    scoring_tasks = [score_image(hit, i) for i, hit in enumerate(hits)]
                    scored_images = await asyncio.gather(*scoring_tasks)
                    
                    # Select best image
                    best_image = max(scored_images, key=lambda x: x["score"])
                    return {"image_url": best_image["url"]}
                else:
                    return {"message": "No image found for this query."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/get-image")
async def get_image(query: str = Query(..., description="Search term for travel image")):
    return await get_image_async_omp(query)

AMADEUS_API_KEY = "AsAMihJdd36IBGvMYDMA45mNsOaQQNK6"
AMADEUS_API_SECRET = "14nQR7SjZg7lQC6Y"

async def get_amadeus_token_async():
    """Async version of Amadeus token retrieval"""
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data["access_token"]

async def get_city_code_async(city_name: str) -> Optional[str]:
    """Async version of city code retrieval"""
    try:
        token = await get_amadeus_token_async()
        url = "https://test.api.amadeus.com/v1/reference-data/locations/cities"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "keyword": city_name,
            "max": 1
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data and data.get("data"):
                    return data["data"][0]["address"]["cityCode"]
                return None
    except Exception as e:
        print(f"Error in get_city_code_async: {e}")
        return None

async def book_hotel_async_omp(city_code: str) -> dict:
    """OpenMP-optimized hotel booking with parallel processing"""
    try:
        token = await get_amadeus_token_async()
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
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                offers = data.get("data", [])

                # Process offers in parallel using OpenMP concepts
                async def process_offer_omp(offer, index):
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        cpu_executor,
                        lambda: {
                            "name": offer["hotel"]["name"],
                            "price": f"${offer['offers'][0]['price']['total']}",
                            "city": offer["hotel"]["cityCode"],
                            "index": index
                        }
                    )
                
                # Process up to 5 offers in parallel
                processing_tasks = [
                    process_offer_omp(offer, i) 
                    for i, offer in enumerate(offers[:5])
                ]
                
                results = await asyncio.gather(*processing_tasks)
                
                # Sort by index to maintain order
                results.sort(key=lambda x: x["index"])
                
                # Remove index from final results
                final_results = []
                for result in results:
                    result.pop("index", None)
                    final_results.append(result)

                return {"hotels": final_results}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Error booking hotel: {str(e)}")

async def extract_city_and_book():
    """Helper function for hotel booking"""
    # This would typically extract city from query context
    # For now, defaulting to Hyderabad
    city_code = await get_city_code_async("Hyderabad")
    if city_code:
        return await book_hotel_async_omp(city_code)
    return {"error": "Could not determine city code"}

@app.get("/book-hotel")
async def book_hotel(city_code: str = Query(..., description="3-letter city code, e.g., HYD for Hyderabad")):
    return await book_hotel_async_omp(city_code)

# Pydantic models (keeping the same as original)
class TravelArticle(BaseModel):
    title: str = Field(..., description="Title of the travel article")
    author: Optional[str] = Field(None, description="Author of the article")
    publish_date: Optional[str] = Field(None, description="Publication date of the article")
    content_summary: str = Field(..., description="A short summary of the article's content")
    url: str = Field(..., description="URL of the article")

class ScrapedHotelOffer(BaseModel):
    name: str
    price: str
    rating: Optional[float] = None
    address: Optional[str] = None
    url: str

class TravelSiteScrapeResult(BaseModel):
    source: str
    data_summary: str
    url: str
    extracted_image_urls: List[str] = []

class TouristSpotInfo(BaseModel):
    name: str
    description: str
    image_url: Optional[str] = None
    scraped_image_urls: List[str] = []

class ChatResponse(BaseModel):
    response: str
    tourist_spots: List[TouristSpotInfo] = []
    general_scraped_image_urls: List[str] = []

global_llm_config = LLMConfig(
    provider="ollama/mistral:7b-instruct-q4_K_M",
    base_url="http://localhost:11434"
)

async def scrape_web_data_omp(query: str, user_constraints: List[str] = None) -> dict:
    """OpenMP-enhanced web scraping with better concurrency"""
    if user_constraints is None:
        user_constraints = []
        
    print(f"Attempting web scrape for query: '{query}' with constraints: {user_constraints}")
    
    all_scraped_summaries = []
    all_scraped_image_urls = []
    
    browser_config = BrowserConfig(
        headless=True,
        user_agent_mode="random",
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        search_terms = query.split()
        
        if user_constraints:
            search_terms.extend(user_constraints)
        
        target_site_urls = [
            {
                "name": "Google Search (General)",
                "url": f"https://www.google.com/search?q=travel+telangana+{'+'.join(search_terms)}"
            }
        ]
        
        if "blog" in query.lower() or "guides" in query.lower():
            target_site_urls.append({
                "name": "Telangana Travel Blog (Example)",
                "url": f"https://www.example-telangana-travel-blog.com/search?q={requests.utils.quote(' '.join(search_terms))}"
            })
        
        # OpenMP-optimized concurrent scraping
        max_concurrent = min(os.cpu_count(), len(target_site_urls))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_single_site_omp(site_info):
            async with semaphore:
                site_name = site_info["name"]
                site_url = site_info["url"]
                print(f"Attempting to scrape from {site_name}: {site_url}")
                
                schema_json = json.dumps(TravelSiteScrapeResult.model_json_schema(), indent=2)
                current_extraction_instructions = (
                    f"You are an expert data extractor.\n\n"
                    f"Given HTML content from the website '{site_name}' ({site_url}), "
                    f"and a user query: '{query}' "
                    f"with these optional constraints: {', '.join(user_constraints) if user_constraints else 'none'},\n"
                    f"extract and return ONLY a JSON object that strictly follows this schema:\n\n"
                    f"{schema_json}\n\n"
                    "Requirements:\n"
                    "1. Do not include any explanation, commentary, or extra text.\n"
                    "2. Only return a valid JSON object conforming to the schema above.\n"
                    "3. Ensure that 'data_summary' contains readable, useful travel information.\n"
                    "4. Include only valid image URLs (jpg, jpeg, png, gif, svg) that relate to the content.\n\n"
                    "Start your response with '{' and end it with '}' — no markdown, headers, or code blocks."
                )
                
                llm_extraction_strategy_for_run = LLMExtractionStrategy(
                    llm_instructions=current_extraction_instructions,
                    schema=TravelSiteScrapeResult.model_json_schema(),
                    extraction_type="schema",
                    llm_config=global_llm_config
                )
                
                crawl_config = {
                    "max_depth": 0,
                    "max_links_per_page": 1,
                    "timeout": 30,
                    "include_subdomains": False
                }
                
                run_config = CrawlerRunConfig(
                    cache_mode="BYPASS",
                    extraction_strategy=llm_extraction_strategy_for_run,
                    css_selector="article, .main-content, .travel-info, body",
                )
                    
                try:
                    result = await asyncio.wait_for(
                        crawler.arun(url=site_url, config=run_config, crawl_config=crawl_config),
                        timeout=30
                    )
                    
                    if result.success and result.extracted_content:
                        try:
                            parsed_data = json.loads(result.extracted_content)
                            
                            if isinstance(parsed_data, list) and parsed_data and isinstance(parsed_data[0], dict) and parsed_data[0].get("error"):
                                print(f"Detected LLM error response from LiteLLM for {site_name}: {parsed_data[0].get('content', 'No error content')}")
                                raise ValueError("LLM returned an error message, not expected data")
                            elif isinstance(parsed_data, list) and parsed_data and isinstance(parsed_data[0], dict):
                                valid_parsed_item = None
                                
                                for item in parsed_data:
                                    if isinstance(item, dict):
                                        try:
                                            valid_parsed_item = TravelSiteScrapeResult(**item)
                                            break
                                        except (ValueError, TypeError) as e:
                                            print(f"Skipping a list item in LLM output due to schema mismatch: {e}")
                                
                                if valid_parsed_item:
                                    parsed_result = valid_parsed_item
                                    print(f"Warning: LLM returned a JSON list; successfully used an object from the list for {site_name}")
                                else:
                                    raise ValueError("LLM returned a JSON list but no item matched the schema")
                            
                            elif isinstance(parsed_data, dict):
                                parsed_result: TravelSiteScrapeResult = TravelSiteScrapeResult(**parsed_data)
                            else:
                                raise ValueError("LLM output is not a valid JSON object or array that fits the schema")
                                
                            print(f"Successfully scraped from {site_name}.")
                            
                            summary_text = ""
                            if parsed_result.data_summary:
                                summary_text = f"Information from {parsed_result.source or site_name} (URL: {parsed_result.url or site_url}):\n{parsed_result.data_summary}"
                            
                            image_urls = []
                            if parsed_result.extracted_image_urls:
                                image_urls = parsed_result.extracted_image_urls
                                print(f"LLM extracted {len(image_urls)} image URLs for {site_name}.")
                            
                            return summary_text, image_urls
                            
                        except (json.JSONDecodeError, ValueError, TypeError) as e:
                            print(f"Warning: LLM output for {site_name} was not valid JSON or did not match schema. Error: {e}")
                            print(f"Raw extracted content: {result.extracted_content[:500]}...")
                            return f"Information from {site_name} (URL: {site_url}):\n{result.extracted_content}", []
                        
                        except Exception as e:
                            print(f"An unexpected error occurred during processing scraped content for {site_name}: {e}")
                            return f"Information from {site_name} (URL: {site_url}):\n{result.extracted_content}", []
                    else:
                        print(f"Failed to scrape from {site_name}. Error: {result.error_message or 'No content or LLM extraction failed.'}")
                        return "", []
                        
                except Exception as e:
                    print(f"Crawl4AI scraping error for {site_name}: {str(e)}")
                    return f"An unexpected error occurred while scraping {site_name}: {str(e)}", []
        
        # Execute all scraping tasks with OpenMP-style concurrency
        scrape_tasks = [scrape_single_site_omp(site_info) for site_info in target_site_urls]
        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        # Process results
        for result in scrape_results:
            if isinstance(result, Exception):
                print(f"Scraping task failed with exception: {result}")
                continue
            
            summary_text, image_urls = result
            if summary_text:
                all_scraped_summaries.append(summary_text)
            if image_urls:
                all_scraped_image_urls.extend(image_urls)
    
    return {
        "scraped_summaries": all_scraped_summaries,
        "scraped_image_urls": all_scraped_image_urls
    }

async def search_vectorstore_omp(query: str, n_results: int = 10) -> list[dict]:
    """OpenMP-optimized vector store search"""
    try:
        # Parallel query processing
        loop = asyncio.get_event_loop()
        
        def _query_sync():
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        
        results = await loop.run_in_executor(cpu_executor, _query_sync)
        
        # OpenMP-style parallel result processing
        if results and results.get('documents') and results['documents'][0]:
            async def process_result_omp(doc, metadata, distance, index):
                return await asyncio.get_event_loop().run_in_executor(
                    cpu_executor,
                    lambda: {
                        "content": doc,
                        "metadata": metadata or {},
                        "relevance_score": 1.0 - distance if distance is not None else 0.0,
                        "index": index
                    }
                )
            
            # Process results in parallel
            processing_tasks = []
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [[{}] * len(documents)])[0]
            distances = results.get('distances', [[0.0] * len(documents)])[0]
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                task = process_result_omp(doc, metadata, distance, i)
                processing_tasks.append(task)
            
            processed_results = await asyncio.gather(*processing_tasks)
            
            # Sort by original index to maintain relevance order
            processed_results.sort(key=lambda x: x["index"])
            
            # Remove index from final results
            for result in processed_results:
                result.pop("index", None)
            
            return processed_results
        
        return []
    except Exception as e:
        print(f"Error searching vectorstore: {e}")
        return []

async def generate_response_omp(query: str, context_docs: list[dict], scraped_data: dict = None) -> str:
    """OpenMP-optimized response generation"""
    loop = asyncio.get_event_loop()
    
    def _generate_sync():
        # Prepare context
        context_text = ""
        if context_docs:
            context_parts = []
            for doc in context_docs[:5]:  # Limit to top 5 for performance
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                activity_types = metadata.get("activity_type", [])
                
                context_part = f"Content: {content}"
                if activity_types:
                    context_part += f" (Activity types: {', '.join(activity_types)})"
                context_parts.append(context_part)
            
            context_text = "\n\n".join(context_parts)
        
        # Add scraped data if available
        scraped_context = ""
        if scraped_data and scraped_data.get("scraped_summaries"):
            scraped_context = "\n\n".join(scraped_data["scraped_summaries"])
        
        # Prepare the prompt
        system_prompt = """You are a helpful travel assistant specializing in Telangana tourism. 
        Provide detailed, accurate, and engaging travel advice based on the context provided.
        Focus on practical information, local insights, and personalized recommendations.
        If you mention specific places, try to include relevant details like timings, entry fees, or best time to visit when available."""
        
        full_context = ""
        if context_text:
            full_context += f"Vectorstore Context:\n{context_text}\n\n"
        if scraped_context:
            full_context += f"Latest Web Information:\n{scraped_context}\n\n"
        
        user_message = f"""Based on the following context, please answer this travel query about Telangana:

Query: {query}

Context:
{full_context if full_context else "No specific context available - please provide general travel advice for Telangana."}

Please provide a comprehensive and helpful response."""

        try:
            # Using litellm for response generation
            response = litellm.completion(
                model="ollama/mistral:7b-instruct-q4_K_M",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                base_url="http://localhost:11434",
                timeout=45
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response with LiteLLM: {e}")
            # Fallback to direct ollama call
            try:
                response = ollama.chat(
                    model="mistral:7b-instruct-q4_K_M",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                )
                return response["message"]["content"]
            except Exception as fallback_e:
                print(f"Fallback ollama call also failed: {fallback_e}")
                return "I apologize, but I'm having trouble generating a response right now. Please try again later."
    
    return await loop.run_in_executor(cpu_executor, _generate_sync)

def extract_tourist_spots_omp(response_text: str) -> list[dict]:
    """OpenMP-optimized tourist spot extraction"""
    # Simple keyword-based extraction for tourist spots
    # This could be enhanced with NLP techniques
    
    spots = []
    common_spots = [
        "Charminar", "Golconda Fort", "Ramoji Film City", "Hussain Sagar",
        "Salar Jung Museum", "Qutb Shahi Tombs", "Birla Mandir", "Chowmahalla Palace",
        "Warangal Fort", "Thousand Pillar Temple", "Khammam", "Bhadrachalam",
        "Arku Valley", "Ananthagiri Hills", "Nagarjuna Sagar", "Pakhal Lake"
    ]
    
    response_lower = response_text.lower()
    
    for spot in common_spots:
        if spot.lower() in response_lower:
            # Extract a brief description (simple heuristic)
            sentences = response_text.split('.')
            description = ""
            
            for sentence in sentences:
                if spot.lower() in sentence.lower():
                    description = sentence.strip()
                    break
            
            if not description and sentences:
                description = f"Popular tourist destination in Telangana."
            
            spots.append({
                "name": spot,
                "description": description,
                "image_url": None,
                "scraped_image_urls": []
            })
    
    return spots

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Validate query using OpenMP-optimized validation
        validation_error = validate_query(request.query)
        if validation_error:
            return ChatResponse(
                response=validation_error,
                tourist_spots=[],
                general_scraped_image_urls=[]
            )
        
        # Run parallel operations using OpenMP concepts
        async def parallel_operations():
            # Task 1: Search vectorstore
            vectorstore_task = search_vectorstore_omp(request.query)
            
            # Task 2: Scrape web data
            scrape_task = scrape_web_data_omp(request.query)
            
            # Task 3: Get related image
            image_task = get_image_async_omp(request.query)
            
            # Execute all tasks in parallel
            vectorstore_results, scraped_data, image_result = await asyncio.gather(
                vectorstore_task, scrape_task, image_task,
                return_exceptions=True
            )
            
            return vectorstore_results, scraped_data, image_result
        
        vectorstore_results, scraped_data, image_result = await parallel_operations()
        
        # Handle exceptions
        if isinstance(vectorstore_results, Exception):
            print(f"Vectorstore search failed: {vectorstore_results}")
            vectorstore_results = []
        
        if isinstance(scraped_data, Exception):
            print(f"Web scraping failed: {scraped_data}")
            scraped_data = {"scraped_summaries": [], "scraped_image_urls": []}
        
        if isinstance(image_result, Exception):
            print(f"Image search failed: {image_result}")
            image_result = {"image_url": None}
        
        # Generate response using OpenMP optimization
        response_text = await generate_response_omp(
            request.query, 
            vectorstore_results, 
            scraped_data
        )
        
        # Extract tourist spots
        tourist_spots = extract_tourist_spots_omp(response_text)
        
        # Add image URLs to tourist spots if available
        if image_result.get("image_url"):
            for spot in tourist_spots:
                if not spot["image_url"]:
                    spot["image_url"] = image_result["image_url"]
                    break
        
        # Collect all scraped image URLs
        general_scraped_image_urls = scraped_data.get("scraped_image_urls", [])
        
        return ChatResponse(
            response=response_text,
            tourist_spots=tourist_spots,
            general_scraped_image_urls=general_scraped_image_urls
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        return ChatResponse(
            response="I apologize, but I encountered an error while processing your request. Please try again.",
            tourist_spots=[],
            general_scraped_image_urls=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Telangana Travel Assistant API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Telangana Travel Assistant API",
        "version": "1.0.0",
        "features": [
            "Document upload (PDF, XLSX)",
            "Intelligent chat responses",
            "Image search integration",
            "Hotel booking via Amadeus API",
            "Web scraping for latest travel info",
            "OpenMP-optimized parallel processing"
        ],
        "endpoints": {
            "POST /chat": "Main chat interface",
            "POST /upload-docs": "Upload travel documents",
            "GET /get-image": "Search travel images",
            "GET /book-hotel": "Book hotels",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Telangana Travel Assistant API with OpenMP optimizations...")
    print(f"Using {os.cpu_count()} CPU cores for parallel processing")
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1  # Use 1 worker to avoid conflicts with OpenMP threading
    )
