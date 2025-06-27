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

os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_PROVIDER"] = "ollama"
os.environ["LITELLM_MODEL"] = "mistral"
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["LITELLM_TIMEOUT"] = "45" 
os.environ["LITELLM_API_BASE"] = os.environ["OLLAMA_API_BASE"]

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        count = 0
        for text in texts:
            print(f"Embedding chunk {count + 1} : {text}")
            response = ollama.embeddings(model="mistral", prompt=text)
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
                doc_metadata["activity_type"] = doc_metadata.get("activity_type",[]) + ["family","kids_friendly"]
            
            if "adventure" in doc.page_content.lower() or "trekking" in doc.page_content.lower():
                doc_metadata["activity_type"] = doc_metadata.get("activity_type",[]) + ["adventure"]
            
            if "water" in doc.page_content.lower() or "lake" in doc.page_content.lower() or "river" in doc.page_content.lower() or "sea" in doc.page_content.lower() or "beaches" in doc.page_content.lower():
                doc_metadata["activity_type"] = doc_metadata.get("activity_type",[]) + ["water_activities"]
               
            if "history" in doc.page_content.lower() or "fort" in doc.page_content.lower() or "temple" in doc.page_content.lower():
                doc_metadata["activity_type"] = doc_metadata.get("activity_type",[]) + ["historic_study_tour"]
            
            if "activity_type" in doc_metadata:
                doc_metadata["activity_type"] = list(set(doc_metadata["activity_type"]))
            
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
        response = requests.get(url, headers = headers, params = params)
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
        response = requests.get(url, headers=headers, params=params)
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
    except requests.exceptions.RequestException as req_e:
        raise HTTPException(status_code = response.status_code if 'response' in locals() else 500, detail = f"Amadeus API error: {req_e}")
    except Exception as e:
        #return {"error": str(e)}
        print("Error:", str(e))
        raise HTTPException(status_code = 500, detail = f"Error booking hotel: {str(e)}")

class TravelArticle(BaseModel):
    title: str = Field(..., description = "Title of the travel article")
    author: Optional[str] = Field(None, description = "Author of the article")
    publish_date: Optional[str] = Field(None, description = "Publication date of the article")
    content_summary: str = Field(..., description = "A short summary of the article's content")
    url: str = Field(..., description = "URL of the article")

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
    provider = "ollama/mistral",
    base_url = "http://localhost:11434"
)

async def scrape_web_data(query: str, user_constraints: List[str] = None) -> dict:
    if user_constraints is None:
        user_constraints = []
        
    print(f"Attempting web scrape for query: '{query}' with constrains: {user_constraints}")
    
    all_scraped_summaries = []
    all_scraped_image_urls = []
    
    browser_config = BrowserConfig(
        headless = True,
        user_agent_mode = "random",
    )
    
    async with AsyncWebCrawler(config = browser_config) as crawler:
        search_terms = query.split()
        
        if user_constraints:
            search_terms.extend(user_constraints)
        
        target_site_urls = []
    
        target_site_urls.append({
            "name": "Google Search (General)",
            "url": f"https://www.google.com/search?q=travel+telangana+{'+'.join(search_terms)}"
        })
        
        if "blog" in query.lower() or "guides" in query.lower():
            target_site_urls.append({
                "name": "Telangana Travel Blog (Example)",
                "url": f"https://www.example-telangana-travel-blog.com/search?q={requests.utils.quote(' '.join(search_terms))}"
            })
        
        for site_info in target_site_urls:
            site_name = site_info["name"]
            site_url = site_info["url"]
            print(f"Attempting to scrape from {site_name}: {site_url}")
            
            schema_json = json.dumps(TravelSiteScrapeResult.model_json_schema(), indent = 2)
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
                llm_instructions = current_extraction_instructions,
                schema = TravelSiteScrapeResult.model_json_schema(),
                extraction_type = "schema",
                llm_config = global_llm_config
            )
            
            crawl_config = {
                "max_depth": 0,
                "max_links_per_page": 1,
                "timeout": 30,
                "include_subdomains": False
            }
            
            run_config = CrawlerRunConfig(
                cache_mode = "BYPASS",
                extraction_strategy = llm_extraction_strategy_for_run,
                css_selector = "article, .main-content, .travel-info, body",
            )
                
        try:
            result = await asyncio.wait_for(
                crawler.arun(url = site_url, config = run_config, crawl_config = crawl_config),
                timeout = 30
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
                            raise ValueError("LLM returned an JSON list but no item matched the schema")
                    
                    elif isinstance(parsed_data, dict):
                        parsed_result: TravelSiteScrapeResult = TravelSiteScrapeResult(**parsed_data)
                    
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
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    print(f"Warning: LLM output for {site_name} was not valid JSON or did not match schema. Error: {e}")
                    print(f"Raw extracted content: {result.extracted_content[:500]}...") 
                    all_scraped_summaries.append(
                        f"Information from {site_name} (URL: {site_url}):\n{result.extracted_content}")
                
                except Exception as e:
                    print(f"An unexpected error occurred during processing scraped content for {site_name}: {e}")
                    all_scraped_summaries.append(
                        f"Information from {site_name} (URL: {site_url}):\n{result.extracted_content}")
            else:
                print(f"Failed to scrape from {site_name}. Error: {result.error_message or 'No content or LLM extraction failed.'}")
                if result.extracted_content: 
                    print(f"Raw extracted_content (if any): {str(result.extracted_content)[:500]}...")
                if result.error_message:
                    print(f"Crawl4AI Error Message: {result.error_message}")
        except Exception as e:
            print(f"Crawl4AI scraping error for {site_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_scraped_summaries.append(f"An unexpected error occurred while scraping {site_name}: {str(e)}")
        await asyncio.sleep(5)
    
    if all_scraped_summaries or all_scraped_image_urls:
        full_summary = "\n\n".join(set([s.strip() for s in all_scraped_summaries if s.strip()]))
        
        try:
            full_summary = str(full_summary).strip()
            
            embedding = ollama.embeddings(model = "mistral", prompt = full_summary)["embedding"]
            
            if not embedding or not isinstance(embedding, list):
                raise ValueError("Invalid embedding returned for scraped content")
            
            print("💾 Storing the web scraping in the vector db")
            
            uid = str(uuid.uuid4())
            
            collection.add(
                documents = [full_summary],
                embeddings = [embedding],
                metadatas = [{
                    "source":site_name, 
                    "url":site_url
                }],
                ids = [uid]
            )
            print(f"✅ Scrapped summary saved successfully to the vector db from {site_name}")
        except Exception as e:
            print(f"⚠️ Failed to save scraped summary to the vector db {e}")
            
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
    print("Query Received:",query)
    combined_context = ""
    general_scraped_image_urls = []
    
    validation_prompt = f"""
        You are a validation assistant.
        Your task is to check if the following user query is related to Telangana's travel, tourism, or history.
        If the query is related, respond with only: VALID.
        If the query is not related, respond with only: INVALID
        Query: {query}
    """
    
    validation_response = ollama.chat(model="mistral", messages=[{"role": "user", "content": validation_prompt}])
    
    if "INVALID" in validation_response["message"]["content"]:
        return {"response":"🧞🧞🧞 Sorry, I can only answer questions related to Telangana tourism, travel, or history."}
    
    user_constraints = parse_constraints(query)
    print(f"Detected constraints: {user_constraints}")
    
    query_embedding = ollama.embeddings(model = "mistral", prompt = query)["embedding"]
    
    print("Querying vector DB...")
    # Query vector DB for context
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include = ['documents','metadatas','distances']
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
    
    if should_web_scrape:
        print("Initialising web scrape due to no DB context or specific keywords...")
        
        scrape_result = await scrape_web_data(query, user_constraints)
        
        if scrape_result and scrape_result.get("summary"):
            scrape_content = scrape_result["summary"]
            general_scraped_image_urls.extend(scrape_result.get("image_urls",[]))
            combined_context += "Information from web search:\n" + scrape_content + "\n\n"
            print("Web scraped content added to context.")
            if general_scraped_image_urls:
                print(f"Found {len(general_scraped_image_urls)} general image URLs from web scrape")
        else:
            print("Web scraping yielded no relevant content.")
    
    tool_response = None
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
        
        extracted_city_code = None
        try:
            city_llm_response = ollama.chat(model = "mistral", messages=[{"role": "user", "content": city_code_extraction_prompt}])
            raw_extracted_code = city_llm_response["message"]["content"].strip().upper()
            
            cleaned_code = raw_extracted_code.replace(".","").strip()
            
            if len(cleaned_code) == 3 and cleaned_code.isalpha():
                extracted_city_code = cleaned_code
                print(f"LLM extracted direct city code: {extracted_city_code}")
            else:
                print(f"LLM extracted an invalid city code format: '{raw_extracted_code}'. Falling back to N/A")
                extracted_city_code = "N/A"
                
        except Exception as e:
            print(f"Error during city extraction by LLM: {e}")
            extracted_city_code = "N/A"
        
        if extracted_city_code and extracted_city_code != "N/A":
            try:
                print(f"Attempting to book hotel for city code: {extracted_city_code}")
                hotel_response = book_hotel(city_code=extracted_city_code)
                tool_response = "Hotel booking information:\n" + json.dumps(hotel_response, indent=2) + "\n\n"
            except HTTPException as e:
                tool_response = f"Error booking hotel: {e.detail}"
            except Exception as e:
                tool_response = f"An unexpected error occurred during hotel booking: {str(e)}"
        else:
             tool_response = "Please specify a valid city for which you want to book a hotel. Could not determine city code"
    elif "image" in query.lower() or "images" in query.lower():
        image_query_term = query.lower().replace("get image of", "").replace("generate image of", "").replace("show image of", "").strip()
        
        if image_query_term:
            print(f"Attempting to get image for query: {image_query_term}")
            image_response = get_image(query = image_query_term)
            tool_response = "Image search result:\n" + json.dumps(image_response, indent = 2) + "\n\n"
        else:
            tool_response = "Please specify what image you are looking for."
            
    if tool_response:
        combined_context += tool_response
        print("Tool response added to context.")
    
    if not combined_context:
        final_prompt = f"You are a helpful travel assistant. Please answer the following question. If you don't have enough information, state that you cannot provide a complete answer.\n\nQuestion: {query}"
        print("No specific context found. Using general prompt.")
    else:
        constraints_str = ", ".join(user_constraints) if user_constraints else "none specified"
        final_prompt = f"""You are a helpful and knowledgeable travel assistant specializing in Telangana.
Your goal is to provide the most accurate, relevant, and comprehensive answer to the user's travel query.
Carefully consider all the provided information, which may include details from a local knowledge base, real-time web scrapes, and direct tool results.

The user has specified the following preferences or constraints: {constraints_str}. Ensure your recommendations and answers adhere to these constraints as much as possible.

Context:
{combined_context.strip()}

Question: {query}

Most appropriate and correct answer:
"""
        print("Using combined context and explicit constraints for prompt.")
    
    try:
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": final_prompt}])
        return {"response":"🧞🧞🧞 "+response["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")
