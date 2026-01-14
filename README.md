# Ollama RAG Workflow

A small Retrieval-Augmented Generation (RAG) demo that answers patient questions about **Temporomandibular Disorders (TMD)**.

- **Retriever:** ChromaDB (persistent, local)  
- **Embeddings:** `mxbai-embed-large` (Ollama)  
- **Generator:** `phi3:latest` (Ollama)  
- **Source:** MedlinePlus Health Topic — *Temporomandibular Disorders*  
  https://medlineplus.gov/temporomandibulardisorders.html



## 1) Requirements

- macOS / Linux (Windows WSL works)
- Python **3.10+** (tested on 3.12)
- **Ollama** installed and running → https://ollama.com  
  Pull models:
  ```bash
  ollama pull mxbai-embed-large
  ollama pull phi3:latest
  
- python dependencies (installed via requirements.txt)

## 2) Setup

```bash

# Pull Ollama models used in this project
ollama pull mxbai-embed-large
ollama pull phi3:latest

# Create & activate virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

```
## 3) How it Works
Scraping  topic content from MedlinePlus.

Chunking text into overlapping pieces for better embedding retrieval.

Embedding chunks using mxbai-embed-large (Ollama). 

Store embeddings in ChromaDB 

Retrieve & Generate: When a question is asked, retrieves relevant chunks and passes them to phi3:latest for answer generation.

## 4) Running the workflow

```
# 1) Create embeddings from scraped data & store in Chroma
python -m src.chroma_db_store

# 2)  vector DB stored : 
python -m src.chroma_db_peek

# 3) Run user queries with RAG
python -m src.user_query

```
Data will be stored in:
````
./data/ → scraped Markdown files

./chroma/chroma_store/ → vector database

````


