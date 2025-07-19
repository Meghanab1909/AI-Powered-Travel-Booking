# 🧞 Travigenie : Wish.Wander.Repeat 

**Travigenie** is an intelligent, context-aware chatbot built to guide users through Telangana tourism. It combines document-based RAG, live web scraping, vector search, and a hybrid validation mechanism. The project supports both GUI (Streamlit) and CLI interfaces and is optimized for fast, local inference using quantized models like Mistral via Ollama.

---

## 📁 Project Structure

```
Travigenie/
├── Database/                        # Uploaded PDFs/Excels for vector indexing
├── Documentation/                  # Design references, papers, guides
├── performance_optimization_methods/ # Parallelization trials and benchmarks
├── CLI.py                          # Terminal-based chatbot
├── api_test.py                     # Unit/API testing script
├── app.py                          # FastAPI server setup
├── client.py                       # Streamlit GUI frontend
├── data_tagging.py                 # Document chunking + tagging + embedding
├── frontend.zip                    # Optional web frontend (React or HTML)
├── main.py                         # Core controller: validation, scraping, RAG
├── rag_chat.py                     # RAG testing utility
├── readme.md                       # Project documentation (this file)
├── requirements.txt                # Python dependencies
├── run_all.sh                      # Shell script to launch backend
├── rull_all.bat                    # Windows batch launcher
└── validators.json                 # JSON rules for Stage-1 query validation
```

---

## ⚙️ Setup Guide

### 1. System Dependencies

**Linux / WSL**

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv curl
```

**Windows**

* Install [Python](https://www.python.org/downloads/)
* Install [Ollama](https://ollama.com/download)

### 2. Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama Model

```bash
ollama serve
ollama pull mistral:7b-instruct-q4_K_M
```

---

## 🚀 Running the Project

### 🔁 Run All (Unix)

```bash
chmod +x run_all.sh
./run_all.sh
```

### 🧪 Run Backend & Frontend Manually

* **FastAPI Server**

  ```bash
  uvicorn app:app --reload
  ```

* **Streamlit GUI**

  ```bash
  streamlit run client.py
  ```

* **CLI Interface**

  ```bash
  python3 CLI.py
  ```

* **Upload Documents**

  ```bash
  python3 data_tagging.py
  ```

* **Test RAG Response**

  ```bash
  python3 rag_chat.py
  ```

---

## 🔍 Features

* **Hybrid Validation**

  * `validators.json`: fast keyword check
  * fallback to LLM-based semantic filtering

* **Document Indexing & Retrieval**

  * Upload Excel/PDF to `Database/`
  * Chunked and categorized using `data_tagging.py`
  * Stored in `ChromaDB` with semantic tags

* **Real-Time Web Scraping**

  * When confidence is low, scrapes:

    * [Incredible India](https://incredibleindia.org/)
    * [Telangana Tourism](https://telanganatourism.gov.in/)
    * [TripAdvisor](https://tripadvisor.in/)

* **External APIs**

  * **Pixabay** – Image generation
  * **Amadeus** – Hotel booking by city code

* **Parallelization**

  * `performance_optimization_methods/` contains:

    * Threading
    * `joblib`, `asyncio`, `aiohttp`
    * `Numba + OpenMP` trials

---

## 🌐 Future Work

* Session-based query history
* Language support: Telugu, Hindi
* React-based rich frontend (`frontend.zip`)
* Map integration + image gallery
* Auto-summarization of long responses
