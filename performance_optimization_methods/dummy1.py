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
import time
from contextlib import asynccontextmanager

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

# Environment configuration
os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_PROVIDER"] = "ollama"
os.environ["LITELLM_MODEL"] = "mistral:7b-instruct-q4_K_M"
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["LITELLM_TIMEOUT"] = "60"  # Increased timeout
os.environ["LITELLM_API_BASE"] = os.environ["OLLAMA_API_BASE"]

# Timeout configurations
CRAWL_TIMEOUT = 45  # seconds
LLM_TIMEOUT = 30   # seconds
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        count = 0
        for text in texts:
            print(f"Embedding chunk {count + 1} : {text[:100]}...")
            try:
                response = ollama.embeddings(
                    model="mistral:7b-instruct-q4_K_M", 
                    prompt=text,
                    options={"timeout": LLM_TIMEOUT}
                )
                embeddings.append(response["embedding"])
                count += 1
            except Exception as e:
                print(f"Error embedding chunk {count + 1}: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 4096)  # Assuming 4096 dimensions for mistral
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
    if not file.filename.endswith((".pdf", ".xlsx")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Upload PDF or XLSX.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        file_type = suffix[1:]
        documents = process_document(temp_path, file_type)
        for idx, doc in enumerate(documents):
            doc_metadata = doc.metadata.copy()
            
            if "family" in doc.page_content.lower() or "kids" in doc.page_content.lower():
                doc_metadata["activity_type"] = doc_metadata.get("activity_type", []) + ["family", "kids_friendly"]
            
            if "adventure" in doc.page_content.lower() or "trekking" in doc.page_content.lower():
                doc_metadata["activity_type"] = doc_metadata.get("activity_type", []) + ["adventure"]
            
            if "water" in doc.page_content.lower() or "lake" in doc.page_content.lower() or "river" in doc.page_content.lower() or "sea" in doc.page_content.lower() or "beaches" in doc.page_content.lower():
                doc_metadata["activity_type"] = doc_metadata.get("activity_type", []) + ["water_activities"]
               
            if "history" in doc.page_content.lower() or "fort" in doc.page_content.lower() or "temple" in doc.page_content.lower():
                doc_metadata["activity_type"] = doc_metadata.get("activity_type", []) + ["historic_study_tour"]
            
            if "activity_type" in doc_metadata:
                doc_metadata["activity_type"] = list(set(doc_metadata["activity_type"]))
            
            collection.add(
                documents=[doc.page_content],
                ids=[f"{file.filename}_{idx}"],
                metadatas=[doc_metadata],
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        os.remove(temp_path)

    return JSONResponse(content={"message": "Documents uploaded and added to vectorstore successfully"})

class ChatRequest(BaseModel):
    query: str

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

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return {"error": f"{response.status_code} - {response.text}"}

        data = response.json()
        if data["hits"]:
            image_url = data["hits"][0]["webformatURL"]
            return {"image_url": image_url}
        else:
            return {"message": "No image found for this query."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out while fetching image"}
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
    response = requests.post(url, data=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["access_token"]

def get_city_code(city_name: str) -> Optional[str]:
    try:
        token = get_amadeus_token()
        url = "https://test.api.amadeus.com/v1/reference-data/locations/cities"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "keyword": city_name,
            "max": 1
        }
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if data and data.get("data"):
            return data["data"][0]["address"]["cityCode"]
        return None
    except requests.exceptions.RequestException as req_e:
        print(f"Amadeus City Search API error: {req_e}")
        return None
    except Exception as e:
        print(f"Error in get_city_code: {e}")
        return None
        
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
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
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
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timed out while booking hotel")
    except requests.exceptions.RequestException as req_e:
        raise HTTPException(status_code=500, detail=f"Amadeus API error: {req_e}")
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Error booking hotel: {str(e)}")

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
    base_url="http://localhost:11434",
    # timeout=LLM_TIMEOUT
)

async def scrape_web_data(query: str, user_constraints: List[str] = None) -> dict:
    if user_constraints is None:
        user_constraints = []
        
    print(f"Attempting web scrape for query: '{query}' with constraints: {user_constraints}")
    
    all_scraped_summaries = []
    all_scraped_image_urls = []
    
    browser_config = BrowserConfig(
        headless=True,
        user_agent_mode="random",
        timeout=CRAWL_TIMEOUT
    )
    
    crawler = None
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            search_terms = query.split()
            
            if user_constraints:
                search_terms.extend(user_constraints)
            
            target_site_urls = []
        
            # Use more reliable travel sites for Telangana
            target_site_urls.extend([
                {
                    "name": "India Travel Guide",
                    "url": f"https://www.incredibleindia.org/content/incredible-india-v2/en/destinations/telangana.html"
                },
                {
                    "name": "Telangana Tourism",
                    "url": f"https://www.telangana.gov.in/tourism"
                },
                {
                    "name": "Travel Search",
                    "url": f"https://www.tripadvisor.com/Tourism-g297586-Telangana-Vacations.html"
                }
            ])
            
            for site_info in target_site_urls:
                site_name = site_info["name"]
                site_url = site_info["url"]
                print(f"Attempting to scrape from {site_name}: {site_url}")
                
                retry_count = 0
                success = False
                
                while retry_count < MAX_RETRIES and not success:
                    try:
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
                            "timeout": CRAWL_TIMEOUT,
                            "include_subdomains": False
                        }
                        
                        run_config = CrawlerRunConfig(
                            cache_mode="BYPASS",
                            extraction_strategy=llm_extraction_strategy_for_run,
                            css_selector="article, .main-content, .travel-info, .content, body",
                            timeout=CRAWL_TIMEOUT
                        )
                        
                        result = await asyncio.wait_for(
                            crawler.arun(url=site_url, config=run_config, crawl_config=crawl_config),
                            timeout=CRAWL_TIMEOUT + 10
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
                                    parsed_result = TravelSiteScrapeResult(**parsed_data)
                                
                                else:
                                    raise ValueError("LLM output is not a valid JSON object or array that fits the schema")
                                    
                                print(f"Successfully scraped from {site_name}.")
                                
                                if parsed_result.data_summary:
                                    all_scraped_summaries.append(
                                        f"Information from {parsed_result.source or site_name} (URL: {parsed_result.url or site_url}):\n{parsed_result.data_summary}"
                                    )
                                
                                if parsed_result.extracted_image_urls:
                                    all_scraped_image_urls.extend(parsed_result.extracted_image_urls)
                                    print(f"LLM extracted {len(parsed_result.extracted_image_urls)} image URLs for {site_name}.")
                                
                                success = True
                                
                            except (json.JSONDecodeError, ValueError, TypeError) as e:
                                print(f"Warning: LLM output for {site_name} was not valid JSON or did not match schema. Error: {e}")
                                if result.extracted_content:
                                    print(f"Raw extracted content: {result.extracted_content[:500]}...")
                                    # Use raw content as fallback
                                    all_scraped_summaries.append(
                                        f"Information from {site_name} (URL: {site_url}):\n{result.extracted_content[:1000]}"
                                    )
                                    success = True
                                else:
                                    raise e
                        else:
                            error_msg = result.error_message or 'No content or LLM extraction failed.'
                            print(f"Failed to scrape from {site_name} (attempt {retry_count + 1}). Error: {error_msg}")
                            if result.extracted_content:
                                print(f"Raw extracted_content (if any): {str(result.extracted_content)[:500]}...")
                            raise Exception(f"Scraping failed: {error_msg}")
                    
                    except asyncio.TimeoutError:
                        retry_count += 1
                        print(f"Timeout occurred for {site_name} (attempt {retry_count}/{MAX_RETRIES})")
                        if retry_count < MAX_RETRIES:
                            print(f"Retrying in {RETRY_DELAY} seconds...")
                            await asyncio.sleep(RETRY_DELAY)
                        else:
                            print(f"Max retries reached for {site_name}")
                            all_scraped_summaries.append(f"Timeout occurred while scraping {site_name} after {MAX_RETRIES} attempts")
                    
                    except Exception as e:
                        retry_count += 1
                        print(f"Crawl4AI scraping error for {site_name} (attempt {retry_count}/{MAX_RETRIES}): {str(e)}")
                        if retry_count < MAX_RETRIES:
                            print(f"Retrying in {RETRY_DELAY} seconds...")
                            await asyncio.sleep(RETRY_DELAY)
                        else:
                            print(f"Max retries reached for {site_name}")
                            all_scraped_summaries.append(f"An error occurred while scraping {site_name}: {str(e)}")
                
                # Add delay between sites to avoid overwhelming servers
                await asyncio.sleep(RETRY_DELAY)
    
    except Exception as e:
        print(f"Critical error in web scraping: {e}")
        traceback.print_exc()
        return {
            "summary": f"Web scraping encountered a critical error: {str(e)}",
            "image_urls": []
        }
    
    if all_scraped_summaries or all_scraped_image_urls:
        full_summary = "\n\n".join(set([s.strip() for s in all_scraped_summaries if s.strip()]))
        
        try:
            full_summary = str(full_summary).strip()
            
            if full_summary:
                embedding = ollama.embeddings(
                    model="mistral:7b-instruct-q4_K_M", 
                    prompt=full_summary[:2000],  # Limit prompt length
                    options={"timeout": LLM_TIMEOUT}
                )["embedding"]
                
                if not embedding or not isinstance(embedding, list):
                    raise ValueError("Invalid embedding returned for scraped content")
                
                print("💾 Storing the web scraping in the vector db")
                
                uid = str(uuid.uuid4())
                
                collection.add(
                    documents=[full_summary],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": "web_scrape", 
                        "url": "multiple_sources",
                        "timestamp": str(time.time())
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
            "summary": "No relevant information could be scraped from the web at this time.",
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
    print("Query Received:",query)
    combined_context = ""
    general_scraped_image_urls = []
    
    with open("validators.json") as f:
        validators = json.load(f)
    
    if(len(query.strip()) < validators["min_query_length"]):
        return {"response":"🧞🧞🧞 Sorry, I am not unable to process this query. Query too short. Try again."}
    
    if(any(block_word in query.lower() for block_word in validators["block_keywords"])):
        return {"response":"🧞🧞🧞 Sorry, I am not unable to process this query. Query contains blocked keywords. Try again."}
    
    if not any(keyword in query.lower() for keyword in validators["telangana_keywords"]):
        return {"response":"🧞🧞🧞 Sorry, I am not unable to process this query. Since the query is not related to Telangana's travel, tourism, or history. Try again."}
        
    validation_prompt = f"""
        You are a validation assistant.
        Your task is to check if the following user query is related to Telangana's travel, tourism, or history.
        If the query is related, respond with only: VALID.
        If the query is not related, respond with only: INVALID
        Query: {query}
    """
    
    validation_response = ollama.chat(model="mistral:7b-instruct-q4_K_M", messages=[{"role": "user", "content": validation_prompt}])
    
    if "INVALID" in validation_response["message"]["content"]:
        return {"response":"🧞🧞🧞 Sorry, I can only answer questions related to Telangana tourism, travel, or history."}
    
    user_constraints = parse_constraints(query)
    print(f"Detected constraints: {user_constraints}")
    
    try:
        query_embedding = ollama.embeddings(
            model="mistral:7b-instruct-q4_K_M", 
            prompt=query,
            options={"timeout": LLM_TIMEOUT}
        )["embedding"]
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        query_embedding = None
    
    print("Querying vector DB...")
    # Query vector DB for context
    db_context_docs = []
    if query_embedding:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results and results["documents"] and any(results["documents"][0]):
                for i, doc_content in enumerate(results["documents"][0]):
                    doc_metadata = results["metadatas"][0][i]
                    
                    doc_activity_types = doc_metadata.get("activity_type")
                    if doc_activity_types is None or not user_constraints or any(uc in doc_activity_types for uc in user_constraints):
                        db_context_docs.append(doc_content)
        except Exception as e:
            print(f"Error querying vector database: {e}")
            results = None
    
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
        or (results and results.get("distances") and results["distances"][0] and results["distances"][0][0] > 0.6)
    )
    
    if should_web_scrape:
        print("Initializing web scrape due to no DB context or specific keywords...")
        
        try:
            scrape_result = await asyncio.wait_for(
                scrape_web_data(query, user_constraints), 
                timeout=CRAWL_TIMEOUT * 2
            )
            
            if scrape_result and scrape_result.get("summary"):
                scrape_content = scrape_result["summary"]
                general_scraped_image_urls.extend(scrape_result.get("image_urls", []))
                combined_context += "Information from web search:\n" + scrape_content + "\n\n"
                print("Web scraped content added to context.")
                if general_scraped_image_urls:
                    print(f"Found {len(general_scraped_image_urls)} general image URLs from web scrape")
            else:
                print("Web scraping yielded no relevant content.")
        except asyncio.TimeoutError:
            print("Web scraping timed out completely")
            combined_context += "Web scraping timed out. Using available local knowledge.\n\n"
        except Exception as e:
            print(f"Web scraping failed: {e}")
            combined_context += f"Web scraping encountered an error: {str(e)}\n\n"
    
    # If no context found, provide a fallback message
    if not combined_context.strip():
        combined_context = "I don't have specific information about this topic in my knowledge base."
    
    # Create the prompt for LLM
    system_prompt = """You are a knowledgeable travel assistant specializing in Telangana tourism. 
    Your role is to provide helpful, accurate, and engaging information about travel destinations, 
    attractions, accommodations, and activities in Telangana.

    When responding:
    1. Be friendly and enthusiastic about Telangana tourism
    2. Provide specific, actionable information when possible
    3. If you mention tourist spots, format them clearly
    4. Include practical details like timings, entry fees, or transportation when available
    5. Always prioritize safety and current information
    6. Keep responses concise but informative"""

    user_prompt = f"""
    Context information:
    {combined_context}

    User query: {query}

    Please provide a helpful response based on the context above. If you mention specific tourist spots or attractions, 
    please format them as follows at the end of your response:

    TOURIST_SPOTS:
    - Spot Name: Description
    - Another Spot: Description

    Response:"""

    try:
        # Generate response using Ollama
        response = ollama.chat(
            model="mistral:7b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"timeout": LLM_TIMEOUT}
        )
        
        llm_response = response["message"]["content"]
        
        # Extract tourist spots from response
        tourist_spots = []
        if "TOURIST_SPOTS:" in llm_response:
            response_parts = llm_response.split("TOURIST_SPOTS:")
            main_response = response_parts[0].strip()
            spots_text = response_parts[1].strip() if len(response_parts) > 1 else ""
            
            # Parse tourist spots
            for line in spots_text.split('\n'):
                if line.strip().startswith('- '):
                    spot_line = line.strip()[2:]  # Remove '- '
                    if ':' in spot_line:
                        spot_name, spot_description = spot_line.split(':', 1)
                        spot_name = spot_name.strip()
                        spot_description = spot_description.strip()
                        
                        # Try to get image for this spot
                        image_url = None
                        try:
                            image_response = get_image(f"{spot_name} Telangana")
                            if isinstance(image_response, dict) and "image_url" in image_response:
                                image_url = image_response["image_url"]
                        except Exception as e:
                            print(f"Error getting image for {spot_name}: {e}")
                        
                        tourist_spots.append(TouristSpotInfo(
                            name=spot_name,
                            description=spot_description,
                            image_url=image_url,
                            scraped_image_urls=[]
                        ))
        else:
            main_response = llm_response
        
        # Create final response
        chat_response = ChatResponse(
            response=main_response,
            tourist_spots=tourist_spots,
            general_scraped_image_urls=general_scraped_image_urls
        )
        
        return chat_response
        
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating response: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)