# src/ingestion/ingest_data.py (formerly create_embeddings.py)

import os
from glob import glob
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore # 🟢 Qdrant Connector
from llama_index.core import VectorStoreIndex # 🟢 LlamaIndex Index
from llama_index.core.schema import Document # 🟢 LlamaIndex Document
from llama_index.core.storage.storage_context import StorageContext # 🟢 Storage Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # 🟢 LlamaIndex Embedding
from dotenv import load_dotenv

# Import all preprocessing functions
from src.data_prep.preprocess_text import process_text_file
from src.data_prep.preprocess_image import process_image_file
from src.data_prep.preprocess_audio import process_audio_file

# --- CONFIGURATION & ENV LOADING ---
ENV_PATH = os.path.join(os.getcwd(), 'config', '.env')
load_dotenv(dotenv_path=ENV_PATH) 

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
# Using the same embedding model for consistency with your previous code
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
PERSIST_DIR = "./data/db" # Directory for LlamaIndex to persist metadata

def initialize_qdrant_client() -> QdrantClient:
    """Initializes and returns the Qdrant client."""
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL not found in environment variables. Check 'config/.env'.")
    try:
        # Qdrant client connection (Qdrant client is standard)
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant at {QDRANT_URL}: {e}")
        print("ACTION REQUIRED: Ensure the Qdrant Docker container is running.")
        exit(1)

def collect_all_documents() -> list[Document]:
    """
    Finds and processes all text, image, and audio files in the data/raw directory.
    """
    # (Document collection logic remains unchanged)
    all_documents = []
    # 1. Process Text Files
    text_files = glob("data/raw/text/*.txt")
    print(f"-> Found {len(text_files)} text files.")
    for filepath in text_files:
        all_documents.extend(process_text_file(filepath))
    
    # 2. Process Image Files
    image_files = glob("data/raw/images/*.png") + glob("data/raw/images/*.jpg") + glob("data/raw/images/*.jpeg")
    print(f"-> Found {len(image_files)} image files.")
    for filepath in image_files:
        all_documents.extend(process_image_file(filepath))
        
    # 3. Process Audio Files
    audio_files = glob("data/raw/audio/*.mp3") + glob("data/raw/audio/*.wav")
    print(f"-> Found {len(audio_files)} audio files.")
    for filepath in audio_files:
        all_documents.extend(process_audio_file(filepath))
        
    print(f"\nCollected a total of {len(all_documents)} documents/chunks for ingestion.")
    return all_documents

def store_documents_qdrant(documents: list[Document]) -> VectorStoreIndex:
    """
    Stores the LlamaIndex Documents into the Qdrant collection using a VectorStoreIndex.
    """
    qdrant_client = initialize_qdrant_client()
    
    # 1. LlamaIndex HuggingFace Embedding Model
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
    # 2. Configure the Qdrant Vector Store
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name=QDRANT_COLLECTION_NAME
    )

    # 3. Configure the Storage Context
    # This tells LlamaIndex where to store its internal mappings/metadata
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. Create the Index (This handles chunking, embedding, and upserting)
    # LlamaIndex will automatically use the IDs (id_) you set in the Document objects
    # to perform an upsert, preventing duplicate ingestion.
    print(f"Creating/Upserting index for {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=embed_model,
        show_progress=True
    )
    
    # 5. Persist LlamaIndex metadata to local disk
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    
    print(f"\nSuccessfully created/upserted index to Qdrant and persisted metadata to {PERSIST_DIR}.")
    return index

if __name__ == "__main__":
    print("--- STARTING MULTI-MODAL DATA INGESTION (LlamaIndex) ---")
    
    # 1. Collect and Process all documents
    all_docs = collect_all_documents()
    
    if all_docs:
        # 2. Store documents in Qdrant and persist index metadata
        store_documents_qdrant(all_docs)
    else:
        print("No documents found in data/raw directories. Ingestion skipped.")