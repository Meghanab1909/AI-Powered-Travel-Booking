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


import atexit
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_PROVIDER"] = "ollama"
os.environ["LITELLM_MODEL"] = "mistral:7b-instruct-q4_K_M"
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["LITELLM_TIMEOUT"] = "45" 
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

# Async embedding function for better concurrency
class AsyncOllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self._async_embed(texts))
    
    async def _async_embed(self, texts: list[str]) -> list[list[float]]:
        """Parallel embedding generation"""
        async def embed_single(text: str, index: int) -> tuple[int, list[float]]:
            loop = asyncio.get_event_loop()
            # Run ollama.embeddings in thread pool to avoid blocking
            response = await loop.run_in_executor(
                cpu_executor,
                lambda: ollama.embeddings(model="mistral:7b-instruct-q4_K_M", prompt=text)
            )
            print(f"Embedding chunk {index + 1}: {text[:50]}...")
            return index, ["embedding"]
        
        # Process embeddings concurrently
        tasks = [embed_single(text, i) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        
        # Sort by original index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])
        return [embedding for _, embedding in sorted_results]

embedding_fn = AsyncOllamaEmbeddingFunction()
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
    
    return await loop.run_in_executor(cpu_executor, _process_sync)

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
        async def process_single_doc(idx: int, doc: Document):
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
            
            # Add to collection - ChromaDB operations are already thread-safe
            collection.add(
                documents=[doc.page_content],
                ids=[f"{file.filename}_{idx}"],
                metadatas=[doc_metadata],
            )
        
        # Process all documents concurrently
        tasks = [process_single_doc(idx, doc) for idx, doc in enumerate(documents)]
        await asyncio.gather(*tasks)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        os.remove(temp_path)

    return JSONResponse(content={"message": "Documents uploaded and added to vectorstore successfully"})

class ChatRequest(BaseModel):
    query: str

PIXABAY_API_KEY = "50909408-7dda07e11adb45c8e645221eb"

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
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return {"error": f"{response.status} - {await response.text()}"}

                data = await response.json()
                if data["hits"]:
                    image_url = data["hits"][0]["webformatURL"]
                    return {"image_url": image_url}
                else:
                    return {"message": "No image found for this query."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/get-image")
async def get_image(query: str = Query(..., description="Search term for travel image")):
    return await get_image_async(query)

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
            async with session.get(url, headers=headers, params=params) as response:
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
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Error booking hotel: {str(e)}")

@app.get("/book-hotel")
async def book_hotel(city_code: str = Query(..., description="3-letter city code, e.g., HYD for Hyderabad")):
    return await book_hotel_async(city_code)

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

async def scrape_web_data(query: str, user_constraints: List[str] = None) -> dict:
    """Enhanced web scraping with better concurrency"""
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
        
        # Process sites concurrently with semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent scrapes
        
        async def scrape_single_site(site_info):
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
        
        # Execute all scraping tasks concurrently
        scrape_tasks = [scrape_single_site(site_info) for site_info in target_site_urls]
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
    
    if all_scraped_summaries or all_scraped_image_urls:
        full_summary = "\n\n".join(set([s.strip() for s in all_scraped_summaries if s.strip()]))
        
        if full_summary:
            try:
                # Run embedding generation in thread pool
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    cpu_executor,
                    lambda: ollama.embeddings(model="mistral:7b-instruct-q4_K_M", prompt=full_summary)["embedding"]
                )
                
                if not embedding or not isinstance(embedding, list):
                    raise ValueError("Invalid embedding returned for scraped content")
                
                print("💾 Storing the web scraping in the vector db")
                
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
                print(f"✅ Scraped summary saved successfully to the vector db")
            except Exception as e:
                print(f"⚠️ Failed to save scraped summary to the vector db: {e}")
                
        return {
            "summary": full_summary,
            "image_urls": list(set(all_scraped_image_urls))
        }
    else:
        print("No summary or image URLs found after all scrape attempts.")
        return {
            "summary": "No relevant information could be scraped from the web.",
            "image_urls": []
        }

def parse_constraints(query: str) -> List[str]:
    constraints = []
    constraint_keywords = {
        "family": ["family", "kids", "children", "child-friendly"],
        "adventure": ["adventure", "trekking", "hiking", "thrill", "safari"],
        "water_activities": ["water activities", "boating", "swimming", "lake", "river", "waterfall"],
        "historic_study_tour": ["historic", "history", "forts", "temples", "museums", "study tour", "heritage"],
        "group_tours": ["group tour", "group travel", "for groups"],
    }
    
    query_lower = query.lower()
    for constraint_type, keywords in constraint_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            constraints.append(constraint_type)
    
    return list(set(constraints))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest = Body(...)):
    query = request.query
    print("Query Received:", query)
    combined_context = ""
    general_scraped_image_urls = []
    
    validation_error = validate_query(query)
    if validation_error:
        print(f"Query validation failed: {validation_error}")
        return {"message": validation_error}
    
    validation_prompt = (
    "You are a validation assistant.\n"
    "Your task is to check if the following user query is related to Telangana's travel, tourism, or history.\n"
    "If the query is related, respond with only: VALID.\n"
    "If the query is not related, respond with only: INVALID\n"
    f"Query: {query}" )

    validation_response = ollama.chat(model="mistral:7b-instruct-q4_K_M", messages=[{"role": "user", "content": validation_prompt}])
    
    if "INVALID" in validation_response["message"]["content"]:
        return {"response":"🧞🧞🧞 Sorry, I can only answer questions related to Telangana tourism, travel, or history."}
    
    print("Query validation passed ✅")
    
    user_constraints = parse_constraints(query)
    print(f"Detected constraints: {user_constraints}")
    
    # Run embedding generation in thread pool
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(
        cpu_executor,
        lambda: ollama.embeddings(model="mistral:7b-instruct-q4_K_M", prompt=query)["embedding"]
    )
    
    print("Querying vector DB...")
    # Query vector DB for context - ChromaDB operations are thread-safe
    results = await loop.run_in_executor(
        io_executor,
        lambda: collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
    )
    
    db_context_docs = []
    if results and results["documents"] and any(results["documents"][0]):
        for i, doc_content in enumerate(results["documents"][0]):
            doc_metadata = results["metadatas"][0][i]
            
            doc_activity_types = doc_metadata.get("activity_type")
            if doc_activity_types is None or not user_constraints or any(uc in doc_activity_types for uc in user_constraints):
                db_context_docs.append(doc_content)
    
    if db_context_docs:
        combined_context += "Information from local knowledge base:\n" + "\n".join([doc for doc in db_context_docs]) + "\n\n"
        print(f"Vector DB context found with {len(db_context_docs)} relevant documents.")
    else:
        print("No direct context found in vector DB or no relevant documents based on constraints")
    
    keywords_for_web_scrape = [
        "current", "latest", "flights", "flight", "buses", "bus", "trains", "train",
        "deal", "deals", "real-time", "weather", "today", "events", "event",
        "reviews", "review", "best place", "best places", "compare", "updated",
        "best price", "best prices", "entry fee", "ticket", "tickets", "fare",
        "open", "closed", "hours", "timings", "crowd", "crowded", "wait time",
        "queue", "safe", "dangerous", "accident", "weather advisory",
        "popular now", "visited most", "hotspot"
    ]
    
    should_web_scrape = (
        not db_context_docs
        or any(kw in query.lower() for kw in keywords_for_web_scrape)
        or (results.get("distances") and results["distances"][0][0] > 0.6)
    )
    
    # Prepare concurrent tasks
    concurrent_tasks = []
    
    # Web scraping task
    if should_web_scrape:
        print("Initializing web scrape due to no DB context or specific keywords...")
        concurrent_tasks.append(("web_scrape", scrape_web_data(query, user_constraints)))
    
    # Tool response tasks
    tool_tasks = []
    
    if "hotel" in query.lower() or "hotels" in query.lower():
        city_code_extraction_prompt = f"""From the following user query, identify the **primary city name** relevant for hotel booking and ONLY return its **3-letter Amadeus City Code**.
        If a specific city is not clearly mentioned, respond ONLY with "N/A".
        Do NOT include state names, country names, conversational filler, explanations, or any punctuation.
        Examples:
        - Query: "book a hotel in Hyderabad" -> City Code: HYD
        - Query: "find hotels near Charminar" -> City Code: HYD
        - Query: "hotels in Telangana" -> City Code: N/A
        - Query: "I need a hotel" -> City Code: N/A

        Query: "{query}"
        City Code:"""
        
        async def extract_city_and_book():
            try:
                city_llm_response = await loop.run_in_executor(
                    cpu_executor,
                    lambda: ollama.chat(model="mistral:7b-instruct-q4_K_M", messages=[{"role": "user", "content": city_code_extraction_prompt}])
                )
                raw_extracted_code = city_llm_response["message"]["content"].strip().upper()
                
                cleaned_code = raw_extracted_code.replace(".", "").strip()
                
                if len(cleaned_code) == 3 and cleaned_code != "N/A":
                    print(f"Extracted city code: {cleaned_code}")
                    return await book_hotel_async(cleaned_code)
                else:
                    print(f"No valid city code extracted from query. Raw response: '{raw_extracted_code}'")
                    return {"message": "Please specify a city name for hotel booking (e.g., 'hotels in Hyderabad')"}
            except Exception as e:
                print(f"Error in hotel booking extraction: {e}")
                return {"error": f"Failed to process hotel booking request: {str(e)}"}
        
        tool_tasks.append(("hotel_booking", extract_city_and_book()))
    
    # Image search task
    image_search_terms = query.lower()
    if any(term in image_search_terms for term in ["image", "photo", "picture", "show me"]):
        tool_tasks.append(("image_search", get_image_async(query)))
    
    # Execute all concurrent tasks
    if concurrent_tasks or tool_tasks:
        all_tasks = []
        task_labels = []
        
        for label, task in concurrent_tasks + tool_tasks:
            all_tasks.append(task)
            task_labels.append(label)
        
        print(f"Executing {len(all_tasks)} concurrent tasks: {task_labels}")
        task_results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(task_results):
            task_label = task_labels[i]
            
            if isinstance(result, Exception):
                print(f"Task {task_label} failed with exception: {result}")
                continue
            
            if task_label == "web_scrape":
                if isinstance(result, dict):
                    scraped_summary = result.get("summary", "")
                    scraped_images = result.get("image_urls", [])
                    
                    if scraped_summary and scraped_summary.strip():
                        combined_context += f"Latest web information:\n{scraped_summary}\n\n"
                        print("✅ Web scraping context added")
                    
                    if scraped_images:
                        general_scraped_image_urls.extend(scraped_images)
                        print(f"✅ Found {len(scraped_images)} images from web scraping")
            
            elif task_label == "hotel_booking":
                if isinstance(result, dict) and "hotels" in result:
                    hotel_info = "\n".join([f"- {hotel['name']}: {hotel['price']}" for hotel in result["hotels"]])
                    combined_context += f"Available Hotels:\n{hotel_info}\n\n"
                    print("✅ Hotel booking information added")
                elif isinstance(result, dict) and ("message" in result or "error" in result):
                    combined_context += f"Hotel Booking: {result.get('message', result.get('error', 'Unknown response'))}\n\n"
            
            elif task_label == "image_search":
                if isinstance(result, dict) and "image_url" in result:
                    general_scraped_image_urls.append(result["image_url"])
                    print("✅ Image search result added")
    
    # Generate final response using LLM
    final_prompt = f"""You are a knowledgeable Telangana travel assistant. Based on the following context and user query, provide a helpful, detailed response about travel in Telangana.

Context:
{combined_context.strip() if combined_context.strip() else "No specific context available."}

User Query: {query}

Provide a comprehensive response that:
1. Directly answers the user's question
2. Includes relevant travel information about Telangana
3. Mentions specific places, activities, or recommendations when appropriate
4. Is friendly and conversational
5. Focuses specifically on Telangana tourism

Response:"""
    
    try:
        llm_response = await loop.run_in_executor(
            cpu_executor,
            lambda: ollama.chat(
                model="mistral:7b-instruct-q4_K_M",
                messages=[{"role": "user", "content": final_prompt}]
            )
        )
        
        final_response = llm_response["message"]["content"]
        print("✅ LLM response generated successfully")
        
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        final_response = "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    # Extract tourist spots from response (simple keyword-based extraction)
    tourist_spots = []
    common_telangana_spots = [
        "Charminar", "Golconda Fort", "Ramoji Film City", "Hussain Sagar", 
        "Birla Mandir", "Salar Jung Museum", "Qutb Shahi Tombs", "Chowmahalla Palace",
        "Warangal Fort", "Thousand Pillar Temple", "Bhadrakali Temple", "Laknavaram Lake",
        "Medak Fort", "Pochampally", "Nagarjuna Sagar", "Alampur", "Mahbubnagar"
    ]
    
    response_lower = final_response.lower()
    for spot in common_telangana_spots:
        if spot.lower() in response_lower:
            # Try to get an image for this spot
            spot_image_result = await get_image_async(f"{spot} Telangana")
            spot_image_url = None
            if isinstance(spot_image_result, dict) and "image_url" in spot_image_result:
                spot_image_url = spot_image_result["image_url"]
            
            tourist_spots.append(TouristSpotInfo(
                name=spot,
                description=f"Famous tourist destination in Telangana",
                image_url=spot_image_url,
                scraped_image_urls=[]
            ))
    
    return ChatResponse(
        response=final_response,
        tourist_spots=tourist_spots,
        general_scraped_image_urls=general_scraped_image_urls
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
