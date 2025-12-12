# 🩺 Medical RAG LLM

## A Retrieval-Augmented Generation System for Medical Knowledge

This project implements a high-performance **Retrieval-Augmented Generation (RAG)** system specialized in answering queries based on private or domain-specific medical knowledge. 

[Image of Retrieval-Augmented Generation system flow]


### ⭐️ Key Features

* **Domain Specialization:** Leverages a custom vector store to ground LLM responses in specific medical literature.
* **Vector Database:** Utilizes **Qdrant** for high-performance vector search and management.
* **LLM Integration:** Powered by the **Gemini API** for embeddings and generative medical query responses.
* **Multimodal Readiness:** Includes structure for text, image, and audio preprocessing pipelines.

---

## 🚀 Getting Started

This guide provides the complete setup required to deploy and run the RAG system locally.

### Prerequisites

Ensure the following tools are installed on your system:

* **Git**
* **Python 3.9+**
* **Docker** (Essential for running the Qdrant service)

### 1. Repository Setup

Clone the project and prepare the environment:

```bash
# Clone the repository
git clone [https://github.com/sudhakar-1104/Medical_RAG_LLM.git](https://github.com/sudhakar-1104/Medical_RAG_LLM.git)
cd Medical_RAG_LLM

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
```
### 2.Dependency Installation

Install all required Python libraries:

```bash
pip install -r requirements.txt

```
### 3.Environment Configuration

Create a file named .env in the project root to store necessary API keys and configuration parameters. Do not commit this
file to Git.

```bash
# --- API & MODEL KEYS ---
GEMINI_API_KEY="[YOUR_GEMINI_API_KEY]" 
ASSEMBLY_API_KEY="[YOUR_ASSEMBLY_API_KEY_IF_NEEDED]" 

# --- QDRANT CONFIGURATION ---
# Default Docker mapping
QDRANT_URL="http://localhost:6333" 
QDRANT_COLLECTION_NAME="Medical_Rsys_Collection" 

# --- LOCAL STORAGE ---
# Path for local data storage
PERSIST_DIR="./db"
```
### 4. Database Deployment (Qdrant)

Start the Qdrant vector database using Docker. This ensures a consistent and isolated environment matching the QDRANT_URL

```bash
docker run -d --name qdrant-rag -p 6333:6333 qdrant/qdrant
```

 💻 Usage

This step processes all raw files, generates embeddings using the Gemini API, and indexes the resulting vectors into the Qdrant collection.


Phase 1: Data Ingestion (Indexing)

1.Data Placement: Place your medical documents (e.g., text, PDF, JSON, etc.) into the ./data/ directory.

2.Execute Ingestion:

```bash
python -m src.ingestion.ingest_data
```
  This script will instantiate the embedding model, connect to Qdrant, and build the vector index.

  
Phase 2: Query Execution

Once the data is successfully indexed, you can run RAG queries against the knowledge base.

A. Command Line Interface (CLI)

Use the query_data.py script for direct, quick-test queries:

```bash
python -m src.query_data
```

🛑 Cleanup

```bash
# Stop the container
docker stop qdrant-rag

# (Optional) Remove the container and associated data
# docker rm qdrant-rag
```

📂 Project Structure

```bash
Medical_RAG_LLM/
├── Config/            # Configuration files and constants
├── data/              # Raw source documents (Input)
├── db/                # Vector store index persistence directory (Output)
├── src/
│   ├── data_prep/     # Scripts for cleaning and transformation (text, image, audio)
│   ├── ingestion/     # Logic for chunking, embedding, and indexing
│   └── query_data.py  # Script for executing RAG search and generation
├── .env               # Environment variables (Ignored by Git)
├── app.py             # Main application entry point (e.g., Web UI)
└── requirements.txt   # Project dependencies
```

### Launching Services (API and UI)

You need two terminal windows running concurrently:

Terminal 1: Start the Backend API (FastAPI)
Run the API server (default port: 8000):

```bash
# Run from project root
uvicorn api_server:app --reload
```

This service handles all communication between the web browser and the RAG logic.

Terminal 2: Start the Frontend Server (UI)
Run a simple server to host the HTML/JS frontend (default port: 5500):
```bash

cd frontend
python -m http.server 5500
```

Access the Dashboard
Open your web browser and navigate to:

```bash
http://localhost:5500/
```



🧑‍💻 Author

1.Sudhakar - sudhakar-1104

2.Shreya - shreyy004




  




















