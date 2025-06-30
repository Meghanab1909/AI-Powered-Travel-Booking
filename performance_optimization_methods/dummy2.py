#2mins

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
import atexit
import logging

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Use environment variables for sensitive data
os.environ["LITELLM_LOG"] = os.getenv("LITELLM_LOG", "DEBUG")
os.environ["LITELLM_PROVIDER"] = os.getenv("LITELLM_PROVIDER", "ollama")
os.environ["LITELLM_MODEL"] = os.getenv("LITELLM_MODEL", "mistral:7b-instruct-q4_K_M")
os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
os.environ["LITELLM_TIMEOUT"] = os.getenv("LITELLM_TIMEOUT", "45")
os.environ["LITELLM_API_BASE"] = os.environ["OLLAMA_API_BASE"]

# Thread pool for CPU-bound operations
cpu_executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on CPU cores
io_executor = ThreadPoolExecutor(max_workers=20)  # Higher for I/O operations

# Cleanup function for thread pools
def cleanup_executors():
    """Clean up thread pool executors"""
    logger.info("Shutting down thread pool executors...")
    cpu_executor.shutdown(wait=True)
    io_executor.shutdown(wait=True)
    logger.info("Thread pool executors shut down successfully")

# Register cleanup function
atexit.register(cleanup_executors)

def load_validators():
    rules = "please follow the rules :)"
    try:
        with open("validators.json", "r") as f:
            print("Reading json file.")
            return json.load(f)
    except FileNotFoundError:
        print("Warning: validators.json not found. Using default validation rules.")
        return rules 
    except json.JSONDecodeError as e:
        print(f"Error parsing validators.json: {e}. Using default validation rules.")
        return rules 

def validate_query(query: str) -> Optional[str]:
    """
    Validates the user query based on predefined rules.
    Returns None if validation passes, or error message string if validation fails.
    """
    rules = load_validators()
    
    # Check minimum query length
    if len(query.strip()) < rules["min_query_length"]:
        return "Please enter a more detailed question."
    
    # Check for blocked keywords
    query_lower = query.lower()
    if any(bad_word in query_lower for bad_word in rules["block_keywords"]):
         return "Your query contains inappropriate content."
    
    # Check for Telangana-related keywords
    if not any(keyword in query_lower for keyword in rules["telangana_keywords"]):
        return "🙏 Sorry, I specialize only in travel queries related to Telangana."
    
    return None  # Validation passed

# Fixed Async embedding function
class AsyncOllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self._async_embed(texts))
    
    async def _async_embed(self, texts: list[str]) -> list[list[float]]:
        """Parallel embedding generation with batching"""
        BATCH_SIZE = 5  # Process in batches to avoid overwhelming the model
        
        async def embed_batch(batch_texts, start_index):
            async def embed_single(text: str, index: int) -> tuple[int, list[float]]:
                try:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        cpu_executor,
                        lambda: ollama.embeddings(model=os.getenv("LITELLM_MODEL", "mistral:7b-instruct-q4_K_M"), prompt=text)
                    )
                    logger.info(f"Embedding chunk {start_index + index + 1}: {text[:50]}...")
                    return start_index + index, response["embedding"]
                except Exception as e:
                    logger.error(f"Error generating embedding for text {start_index + index + 1}: {e}")
                    # Return zero embedding as fallback
                    return start_index + index, [0.0] * 4096  # Typical embedding dimension
            
            tasks = [embed_single(text, i) for i, text in enumerate(batch_texts)]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process in batches
        all_results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            try:
                batch_results = await embed_batch(batch, i)
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch embedding failed: {result}")
                        all_results.append((len(all_results), [0.0] * 4096))
                    else:
                        all_results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Add fallback embeddings for the batch
                for j in range(len(batch)):
                    all_results.append((i + j, [0.0] * 4096))
        
        # Sort by original index to maintain order
        sorted_results = sorted(all_results, key=lambda x: x[0])
        return [embedding for _, embedding in sorted_results]

embedding_fn = AsyncOllamaEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(
    name="telangana_travel",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

app = FastAPI(title="Telangana Travel Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    cleanup_executors()

async def process_document_async(file_path: str, file_type: str) -> list[Document]:
    """Async wrapper for document processing"""
    loop = asyncio.get_event_loop()
    
    def _process_sync():
        try:
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
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    return await loop.run_in_executor(cpu_executor, _process_sync)

def _extract_metadata_sync(doc):
    """Extract metadata from document synchronously"""
    try:
        doc_metadata = doc.metadata.copy()
        content_lower = doc.page_content.lower()
        activity_types = []
        
        if any(word in content_lower for word in ["family", "kids"]):
            activity_types.extend(["family", "kids_friendly"])
        if any(word in content_lower for word in ["adventure", "trekking"]):
            activity_types.append("adventure")
        if any(word in content_lower for word in ["water", "lake", "river", "sea", "beaches"]):
            activity_types.append("water_activities")
        if any(word in content_lower for word in ["history", "fort", "temple"]):
            activity_types.append("historic_study_tour")
        
        if activity_types:
            doc_metadata["activity_type"] = list(set(activity_types))
        return doc_metadata
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
        return doc.metadata.copy()

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
        documents = await process_document_async(temp_path, file_type)
        
        # Process documents in parallel
        async def process_document_batch(documents, filename):
            # Parallel metadata processing
            async def extract_metadata(doc):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(cpu_executor, _extract_metadata_sync, doc)
            
            # Process metadata in parallel
            metadata_tasks = [extract_metadata(doc) for doc in documents]
            all_metadata = await asyncio.gather(*metadata_tasks, return_exceptions=True)
            
            # Handle any exceptions in metadata extraction
            processed_metadata = []
            for i, metadata in enumerate(all_metadata):
                if isinstance(metadata, Exception):
                    logger.warning(f"Metadata extraction failed for doc {i}: {metadata}")
                    processed_metadata.append(documents[i].metadata.copy())
                else:
                    processed_metadata.append(metadata)
            
            # Batch ChromaDB operation
            batch_docs = [doc.page_content for doc in documents]
            batch_ids = [f"{filename}_{idx}" for idx in range(len(documents))]
            
            try:
                collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=processed_metadata
                )
            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB: {e}")
                raise HTTPException(status_code=500, detail=f"Error storing documents: {str(e)}")
        
        await process_document_batch(documents, file.filename)
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return JSONResponse(content={"message": "Documents uploaded and added to vectorstore successfully"})

class ChatRequest(BaseModel):
    query: str

# API Keys - Use environment variables for security
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "50909408-7dda07e11adb45c8e645221eb")
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY", "AsAMihJdd36IBGvMYDMA45mNsOaQQNK6")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET", "14nQR7SjZg7lQC6Y")

async def get_image_async(query: str) -> dict:
    """Async version of get_image using aiohttp"""
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "image_type": "photo",
            "per_page": 3
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    return {"error": f"{response.status} - {await response.text()}"}

                data = await response.json()
                if data["hits"]:
                    image_url = data["hits"][0]["webformatURL"]
                    return {"image_url": image_url}
                else:
                    return {"message": "No image found for this query."}
    except Exception as e:
        logger.error(f"Error fetching image: {e}")
        return {"error": str(e)}

@app.get("/get-image")
async def get_image(query: str = Query(..., description="Search term for travel image")):
    return await get_image_async(query)

async def get_amadeus_token_async():
    """Async version of Amadeus token retrieval"""
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                data = await response.json()
                return data["access_token"]
    except Exception as e:
        logger.error(f"Error getting Amadeus token: {e}")
        raise

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
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data and data.get("data"):
                    return data["data"][0]["address"]["cityCode"]
                return None
    except Exception as e:
        logger.error(f"Error in get_city_code_async: {e}")
        return None

async def book_hotel_async(city_code: str) -> dict:
    """Async version of hotel booking"""
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
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
                data = await response.json()
                offers = data.get("data", [])

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
        logger.error(f"Error booking hotel: {e}")
        raise HTTPException(status_code=500, detail=f"Error booking hotel: {str(e)}")

@app.get("/book-hotel")
async def book_hotel(city_code: str = Query(..., description="3-letter city code, e.g., HYD for Hyderabad")):
    return await book_hotel_async(city_code)

# Fixed extract_city_and_book function
async def extract_city_and_book(query: str = "Hyderabad") -> dict:
    """Extract city from query and book hotels"""
    try:
        # Simple city extraction - you can make this more sophisticated
        common_cities = {
            "hyderabad": "HYD",
            "warangal": "WGC",
            "nizamabad": "NZB",
        }
        
        query_lower = query.lower()
        city_code = None
        
        for city, code in common_cities.items():
            if city in query_lower:
                city_code = code
                break
        
        if not city_code:
            # Try to get city code dynamically
            city_code = await get_city_code_async("Hyderabad")  # Default to Hyderabad
        
        if city_code:
            return await book_hotel_async(city_code)
        else:
            return {"message": "Could not determine city for hotel booking"}
            
    except Exception as e:
        logger.error(f"Error in extract_city_and_book: {e}")
        return {"error": f"Error in hotel booking: {str(e)}"}

# Pydantic models
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
    base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
)

async def scrape_web_data(query: str, user_constraints: List[str] = None) -> dict:
    """Enhanced web scraping with better concurrency and error handling"""
    if user_constraints is None:
        user_constraints = []
        
    logger.info(f"Attempting web scrape for query: '{query}' with constraints: {user_constraints}")
    
    all_scraped_summaries = []
    all_scraped_image_urls = []
    
    browser_config = BrowserConfig(
        headless=True,
        user_agent_mode="random",
    )
    
    try:
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
            
            # Fixed semaphore implementation
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent scrapes
            
            async def scrape_single_site(site_info):
                async with semaphore:
                    site_name = site_info["name"]
                    site_url = site_info["url"]
                    logger.info(f"Attempting to scrape from {site_name}: {site_url}")
                    
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
                                    logger.warning(f"Detected LLM error response from LiteLLM for {site_name}: {parsed_data[0].get('content', 'No error content')}")
                                    raise ValueError("LLM returned an error message, not expected data")
                                elif isinstance(parsed_data, list) and parsed_data and isinstance(parsed_data[0], dict):
                                    valid_parsed_item = None
                                    
                                    for item in parsed_data:
                                        if isinstance(item, dict):
                                            try:
                                                valid_parsed_item = TravelSiteScrapeResult(**item)
                                                break
                                            except (ValueError, TypeError) as e:
                                                logger.warning(f"Skipping a list item in LLM output due to schema mismatch: {e}")
                                    
                                    if valid_parsed_item:
                                        parsed_result = valid_parsed_item
                                        logger.warning(f"LLM returned a JSON list; successfully used an object from the list for {site_name}")
                                    else:
                                        raise ValueError("LLM returned a JSON list but no item matched the schema")
                                
                                elif isinstance(parsed_data, dict):
                                    parsed_result: TravelSiteScrapeResult = TravelSiteScrapeResult(**parsed_data)
                                else:
                                    raise ValueError("LLM output is not a valid JSON object or array that fits the schema")
                                    
                                logger.info(f"Successfully scraped from {site_name}.")
                                
                                summary_text = ""
                                if parsed_result.data_summary:
                                    summary_text = f"Information from {parsed_result.source or site_name} (URL: {parsed_result.url or site_url}):\n{parsed_result.data_summary}"
                                
                                image_urls = []
                                if parsed_result.extracted_image_urls:
                                    image_urls = parsed_result.extracted_image_urls
                                    logger.info(f"LLM extracted {len(image_urls)} image URLs for {site_name}.")
                                
                                return summary_text, image_urls
                                
                            except (json.JSONDecodeError, ValueError, TypeError) as e:
                                logger.warning(f"LLM output for {site_name} was not valid JSON or did not match schema. Error: {e}")
                                logger.debug(f"Raw extracted content: {result.extracted_content[:500]}...")
                                return f"Information from {site_name} (URL: {site_url}):\n{result.extracted_content}", []
                            
                            except Exception as e:
                                logger.error(f"An unexpected error occurred during processing scraped content for {site_name}: {e}")
                                return f"Information from {site_name} (URL: {site_url}):\n{result.extracted_content}", []
                        else:
                            logger.warning(f"Failed to scrape from {site_name}. Error: {result.error_message or 'No content or LLM extraction failed.'}")
                            return "", []
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout occurred while scraping {site_name}")
                        return "", []
                    except Exception as e:
                        logger.error(f"Crawl4AI scraping error for {site_name}: {str(e)}")
                        return f"An unexpected error occurred while scraping {site_name}: {str(e)}", []
            
            # Execute all scraping tasks concurrently
            scrape_tasks = [scrape_single_site(site_info) for site_info in target_site_urls]
            scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            
            # Process results
            for result in scrape_results:
                if isinstance(result, Exception):
                    logger.error(f"Scraping task failed with exception: {result}")
                    continue
                
                summary_text, image_urls = result
                if summary_text:
                    all_scraped_summaries.append(summary_text)
                if image_urls:
                    all_scraped_image_urls.extend(image_urls)
    
    except Exception as e:
        logger.error(f"Error in web scraping: {e}")
        return {
            "summary": "Web scraping encountered an error.",
            "image_urls": []
        }
    
    if all_scraped_summaries or all_scraped_image_urls:
        full_summary = "\n\n".join(set([s.strip() for s in all_scraped_summaries if s.strip()]))
        
        if full_summary:
            try:
                # Run embedding generation in thread pool
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    cpu_executor,
                    lambda: ollama.embeddings(model=os.getenv("LITELLM_MODEL", "mistral:7b-instruct-q4_K_M"), prompt=full_summary)["embedding"]
                )
                
                if not embedding or not isinstance(embedding, list):
                    raise ValueError("Invalid embedding returned for scraped content")
                
                logger.info("💾 Storing the web scraping in the vector db")
                
                uid = str(uuid.uuid4())
                
                # ChromaDB operations are thread-safe
                collection.add(
                    documents=[full_summary],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": "web_scrape",
                        "url": "multiple"
                    }],
                    ids=[uid]
                )
                logger.info(f"✅ Scraped summary saved successfully to the vector db")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save scraped summary to the vector db: {e}")
                
        return {
            "summary": full_summary,
            "image_urls": list(set(all_scraped_image_urls))
        }
    else:
        logger.info("No summary or image URLs found after all scrape attempts.")
        return {
            "summary": "No relevant information could be scraped from the web.",
            "image_urls": []
        }

def parse_constraints(query: str) -> List[str]:
    """Parse user constraints from query"""
    constraints = []
    
    # Extract budget constraints
    if "budget" in query.lower() or "cheap" in query.lower() or "affordable" in query.lower():
        constraints.append("budget-friendly")
    if "luxury" in query.lower() or "premium" in query.lower():
        constraints.append("luxury")
    
    # Extract activity preferences
    if "family" in query.lower() or "kids" in query.lower():
        constraints.append("family-friendly")
    if "adventure" in query.lower() or "trek" in query.lower():
        constraints.append("adventure")
    if "historic" in query.lower() or "heritage" in query.lower():
        constraints.append("historical")
    if "nature" in query.lower() or "wildlife" in query.lower():
        constraints.append("nature")
    
    # Extract duration constraints
    if "weekend" in query.lower() or "2 day" in query.lower():
        constraints.append("short-trip")
    if "week" in query.lower() or "7 day" in query.lower():
        constraints.append("long-trip")
    
    return constraints

async def generate_response_async(query: str, context: str, web_data: str = "") -> str:
    """Generate response using LiteLLM with proper async handling"""
    try:
        system_prompt = """You are a helpful travel assistant specializing in Telangana tourism. 
        Use the provided context and web data to give comprehensive, accurate travel advice.
        Focus on Telangana destinations, culture, food, and travel tips.
        If hotel booking is mentioned, provide helpful information about accommodations.
        Be friendly, informative, and encouraging."""
        
        user_prompt = f"""Query: {query}

Context from documents:
{context}

Additional web information:
{web_data}

Please provide a helpful response about Telangana travel."""

        loop = asyncio.get_event_loop()
        
        # Run LiteLLM completion in thread pool
        response = await loop.run_in_executor(
            io_executor,
            lambda: litellm.completion(
                model=os.getenv("LITELLM_MODEL", "mistral:7b-instruct-q4_K_M"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                timeout=45
            )
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while processing your request. However, based on the available information: {context[:200]}..."

async def extract_tourist_spots(response_text: str, query: str) -> List[TouristSpotInfo]:
    """Extract tourist spots mentioned in the response"""
    try:
        spots = []
        
        # Common Telangana tourist spots
        telangana_spots = {
            "charminar": "Iconic monument and mosque in Hyderabad's old city",
            "golconda fort": "Historic fort complex with stunning architecture",
            "ramoji film city": "World's largest integrated film studio complex",
            "hussain sagar": "Heart-shaped lake with Buddha statue",
            "birla mandir": "Beautiful white marble temple",
            "salar jung museum": "One of India's largest museums",
            "chowmohalla palace": "Magnificent palace of the Nizams",
            "warangal fort": "Medieval fort with impressive gateways",
            "thousand pillar temple": "Ancient temple with intricate carvings",
            "laknavaram lake": "Scenic lake with suspension bridge",
            "bhadrachalam": "Sacred temple town on Godavari river",
            "medak fort": "Historic fort with panoramic views",
            "kolanupaka": "Ancient Jain temple complex"
        }
        
        response_lower = response_text.lower()
        
        for spot_name, description in telangana_spots.items():
            if spot_name in response_lower or spot_name.replace(" ", "") in response_lower:
                # Get image for the spot
                image_data = await get_image_async(f"{spot_name} telangana")
                image_url = image_data.get("image_url", "")
                
                spots.append(TouristSpotInfo(
                    name=spot_name.title(),
                    description=description,
                    image_url=image_url,
                    scraped_image_urls=[]
                ))
        
        return spots[:5]  # Limit to 5 spots
        
    except Exception as e:
        logger.error(f"Error extracting tourist spots: {e}")
        return []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with comprehensive async processing"""
    try:
        query = request.query.strip()
        
        # Validate query
        validation_error = validate_query(query)
        if validation_error:
            return ChatResponse(
                response=validation_error,
                tourist_spots=[],
                general_scraped_image_urls=[]
            )
        
        logger.info(f"Processing query: {query}")
        
        # Extract constraints using LLM
        constraint_prompt = (
            "You are a travel assistant extracting user constraints for Telangana tourism.\n"
            "Given a user query, identify and return relevant constraints like:\n"
            "- district or location\n"
            "- travel type (solo, family, adventure, spiritual, etc.)\n"
            "- preferred time or season\n"
            "- budget\n"
            "- duration (in days)\n"
            "Return only a JSON object with keys: location, travel_type, season, budget, duration.\n"
            f"Query: {query}"
        )

        constraint_response = ollama.chat(
            model="mistral:7b-instruct-q4_K_M",
            messages=[{"role": "user", "content": constraint_prompt}]
        )

        try:
            constraints = json.loads(constraint_response["message"])
        except Exception as e:
            logger.warning(f"Constraint parsing failed: {e}")
            constraints = {}
        
        logger.info(f"Extracted constraints: {constraints}")
        
        # Parse constraints
        constraints = parse_constraints(query)
        
        # Create async tasks for parallel processing
        async def get_vector_context():
            try:
                # Run vector search in thread pool
                loop = asyncio.get_event_loop()
                
                def _vector_search():
                    results = collection.query(
                        query_texts=[query],
                        n_results=5,
                        include=["documents", "metadatas"]
                    )
                    
                    context_parts = []
                    if results["documents"] and results["documents"][0]:
                        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                            source = metadata.get("source", "unknown")
                            context_parts.append(f"[Source: {source}] {doc}")
                    
                    return "\n\n".join(context_parts)
                
                return await loop.run_in_executor(cpu_executor, _vector_search)
                
            except Exception as e:
                logger.error(f"Error in vector search: {e}")
                return "Unable to retrieve context from knowledge base."
        
        # Run parallel tasks
        context_task = asyncio.create_task(get_vector_context())
        scrape_task = asyncio.create_task(scrape_web_data(query, constraints))
        
        # Handle hotel booking if requested
        hotel_info = ""
        if any(word in query.lower() for word in ["hotel", "book", "accommodation", "stay"]):
            try:
                hotel_data = await extract_city_and_book(query)
                if "hotels" in hotel_data:
                    hotel_list = "\n".join([f"- {h['name']}: {h['price']}" for h in hotel_data["hotels"][:3]])
                    hotel_info = f"\n\nHotel Options:\n{hotel_list}"
                elif "message" in hotel_data:
                    hotel_info = f"\n\nHotel Info: {hotel_data['message']}"
            except Exception as e:
                logger.error(f"Hotel booking error: {e}")
                hotel_info = "\n\nHotel booking service is temporarily unavailable."
        
        # Wait for parallel tasks to complete
        context, web_data = await asyncio.gather(context_task, scrape_task, return_exceptions=True)
        
        # Handle exceptions from parallel tasks
        if isinstance(context, Exception):
            logger.error(f"Context retrieval failed: {context}")
            context = "Unable to retrieve context from knowledge base."
        
        if isinstance(web_data, Exception):
            logger.error(f"Web scraping failed: {web_data}")
            web_data = {"summary": "Web scraping temporarily unavailable.", "image_urls": []}
        
        # Combine context
        full_context = context + hotel_info
        web_summary = web_data.get("summary", "")
        
        # Generate response
        response_text = await generate_response_async(query, full_context, web_summary)
        
        # Extract tourist spots and get images in parallel
        spots_task = asyncio.create_task(extract_tourist_spots(response_text, query))
        
        # Get general images for the query
        general_image_task = asyncio.create_task(get_image_async(f"{query} telangana travel"))
        
        tourist_spots, general_image = await asyncio.gather(spots_task, general_image_task, return_exceptions=True)
        
        # Handle exceptions
        if isinstance(tourist_spots, Exception):
            logger.error(f"Tourist spots extraction failed: {tourist_spots}")
            tourist_spots = []
        
        if isinstance(general_image, Exception):
            logger.error(f"General image retrieval failed: {general_image}")
            general_image = {}
        
        # Collect all image URLs
        general_image_urls = []
        if general_image.get("image_url"):
            general_image_urls.append(general_image["image_url"])
        
        # Add web scraped images
        scraped_images = web_data.get("image_urls", [])
        general_image_urls.extend(scraped_images)
        
        # Remove duplicates
        general_image_urls = list(set(general_image_urls))
        
        return ChatResponse(
            response=response_text,
            tourist_spots=tourist_spots,
            general_scraped_image_urls=general_image_urls
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return ChatResponse(
            response="I apologize, but I encountered an error while processing your request. Please try again with a different query about Telangana travel.",
            tourist_spots=[],
            general_scraped_image_urls=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Telangana Travel Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat - Main chat interface",
            "upload": "/upload-docs - Upload travel documents",
            "image": "/get-image - Get travel images",
            "hotel": "/book-hotel - Get hotel information",
            "health": "/health - Health check"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=2,  # Single worker for development
        log_level="info"
    )