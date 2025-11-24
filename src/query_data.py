# src/query_data.py

import os
from typing import List, Dict, Any, Optional
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

# --- CONFIGURATION & ENV LOADING ---
ENV_PATH = os.path.join(os.getcwd(), 'config', '.env')
load_dotenv(dotenv_path=ENV_PATH) 

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERSIST_DIR = "./data/db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
LLM_MODEL = "gemini-2.5-pro" # Use Pro for better adherence to complex persona/constraints

# Global client variables for manual Gemini call
AI_CLIENT = None
GEMINI_CLIENT = None

try:
    import google.genai
    from google.genai.errors import APIError as GeminiAPIError
    from google.genai import types
    if GEMINI_API_KEY:
        GEMINI_CLIENT = google.genai.Client(api_key=GEMINI_API_KEY)
        AI_CLIENT = "gemini"
except ImportError:
    pass 

# ----------------------------------------------------------------------
# 1. LlamaIndex Initialization (Load Index and Retriever)
# ----------------------------------------------------------------------
def initialize_rag_components(llm_model: str, embed_model_name: str):
    """Initializes and returns the LlamaIndex Index and LLM."""
    if not QDRANT_URL or not GEMINI_API_KEY:
        raise ValueError("QDRANT_URL or GEMINI_API_KEY not found in environment variables.")

    # print("Initializing RAG components...") # Removed for clean output
    
    # 1. Initialize Qdrant Client and Vector Store
    qdrant_client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name=QDRANT_COLLECTION_NAME
    )
    
    # 2. Load the LlamaIndex Storage Context and Index
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        persist_dir=PERSIST_DIR
    )
    # Suppress output during index loading
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=HuggingFaceEmbedding(model_name=embed_model_name)
        )
    
    # 3. Configure the LLM for direct API calls
    llm = Gemini(model=llm_model, api_key=GEMINI_API_KEY)
    
    return index, llm


# ----------------------------------------------------------------------
# 2. Adaptive LLM Generation Function
# ----------------------------------------------------------------------
def get_system_instruction(persona: str) -> str:
    """Returns the appropriate system instruction based on the desired persona."""
    if persona == 'DOCTOR':
        # Professional, technical, uses advanced medical terminology
        return (
            "You are a Chief Medical Officer (CMO) performing a rapid, definitive assessment. "
            "Your audience is a medical specialist. Use precise technical terminology (e.g., myalgia, iatrogenic hyperthyroidism, vasogenic edema). "
            "You must state a definite diagnosis, cause, and concrete intervention based on the most likely clinical inference. "
            "FORBIDDEN phrases: 'cannot be established,' 'undetermined,' 'non-specific,' or 'requires further investigation.' "
        )
    else: # Defaulting to PATIENT
        # Layman, simple language, focuses on symptoms and easily understood interventions
        return (
            "You are a caring medical explainer (CMO persona) speaking directly to the patient. "
            "Use clear, simple, and empathetic language. Explain all medical terms using everyday words (e.g., 'Hypothyroidism' should be explained as 'Your body's master gland is running too slow'). "
            "Your output must be reassuring but accurate, focusing on what the patient needs to know and do. "
        )

def call_llm_for_generation(query: str, context: str, source_file: str, persona: str) -> str:
    """
    Calls the Gemini model with an adaptive persona and forces a definite conclusion.
    """
    if GEMINI_CLIENT is None or AI_CLIENT != "gemini":
        return "🛑 ERROR: Gemini client not active. Cannot generate structured response."
    
    system_instruction_core = get_system_instruction(persona)

    system_instruction = (
        system_instruction_core +
        "Your response MUST be strictly structured using Markdown bolding for the three requested section titles below, "
        "and must only contain the structure and content.\n"
        "1. **Clinical Explanation and Summary**\n"
        "2. **Problem/Diagnosis and Cause**\n"
        "3. **Recommended Intervention and Risks**\n"
        "Base your analysis ONLY on the CONTEXT provided."
    )
    
    user_prompt = (
        f"Analyze the context from the file '{source_file}' to address the user request: '{query}'. "
        f"Begin your response immediately with the first structured section title. Adhere strictly to the requested persona.\n\n"
        f"--- CONTEXT FOR ANALYSIS ---\n{context}"
    )
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=LLM_MODEL, 
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction, 
                temperature=0.3, 
            )
        )
        return response.text
        
    except GeminiAPIError as e: 
        return f"\n[Gemini API Error]: Failed to generate response. Error: {e}"
    except Exception as e:
        return f"\n[Gemini Fatal Error]: Failed to generate response. Error: {e}"


# ----------------------------------------------------------------------
# 3. Targeted Retrieval Function (Removed print statements)
# ----------------------------------------------------------------------
def retrieve_targeted_context(index, user_query: str, target_filename: str, top_k: int = 20): # Increased default to 20
    """Retrieves context from the LlamaIndex index, filtered by a specific source file."""

    # Removed print statement for searching/filtering
    
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="source", 
                match=MatchValue(value=target_filename)
            )
        ]
    )
    
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        vector_store_kwargs={"query_filter": qdrant_filter} 
    )
    
    nodes = retriever.retrieve(user_query)
    
    return nodes


# ----------------------------------------------------------------------
# 4. Main Execution Function (Clean Output)
# ----------------------------------------------------------------------
def run_rag_query(user_query: str, file_path: str, persona: str, top_k: int = 20):
    """Executes the full targeted RAG pipeline, strictly enforcing the file filter."""
    
    index, llm = initialize_rag_components(LLM_MODEL, EMBEDDING_MODEL_NAME)
    
    # 1. Normalize path and extract ONLY the filename
    target_filename = os.path.basename(file_path)
    
    # 2. Retrieve Context (LlamaIndex + Qdrant Filter)
    nodes = retrieve_targeted_context(index, user_query, target_filename, top_k)
    
    # 3. CRITICAL FIX: AGGRESSIVELY FILTER NODES
    strictly_filtered_nodes = []
    
    for node in nodes:
        if node.metadata.get('source') == target_filename:
            strictly_filtered_nodes.append(node)
        # Removed the Filter Mismatch warning print statement
    
    nodes = strictly_filtered_nodes # Use the strictly filtered list
    
    # Removed debug prints for hits
    
    if not nodes:
        aggregated_context = f"No context retrieved from the targeted file: {target_filename}. Cannot generate analysis."
    else:
        # 4. Aggregate Context
        context_list = []
        for node in nodes:
            source_file = node.metadata.get('source', 'N/A')
            modality = node.metadata.get('type', 'unknown')
            
            context_list.append(
                f"Source: {source_file} (Type: {modality}, Score: {node.score:.4f})\n"
                f"Content: {node.get_content().strip()}"
            )
        
        aggregated_context = "\n\n--- Retrieved Chunk ---\n\n".join(context_list)
    
    # 5. Generate Structured Summary
    # Removed print statement for calling Gemini
    final_answer = call_llm_for_generation(user_query, aggregated_context, target_filename, persona)

    # 6. Output Results (Cleaned up format)
    print("\n" + "=" * 70)
    print("FINAL STRUCTURED MEDICAL ANALYSIS (Powered by Gemini)")
    print("=" * 70)
    print(final_answer)
    print("-" * 70)
    print(f"📄 Sources Used ({len(nodes)}):\n")
    if nodes:
        for i, node in enumerate(nodes):
            print(f"[{i+1}] Source File: {node.metadata.get('source')} | Score: {node.score:.4f}")
    else:
        print("None.")
    print("=" * 70)


if __name__ == "__main__":
    import contextlib # Import added for clean loading
    
    print("\n--- Multimodal RAG Query Agent (Targeted & Structured) ---")
    print("This agent filters context by file path and generates a structured medical report.")
    print("-" * 50)
    
    try:
        user_query = input("❓ Enter your question (e.g., 'What is the patient diagnosis?'): ").strip()
        file_path = input("📁 Enter target file path (e.g., data/raw/text/Case1.txt): ").strip()
        
        # 🟢 NEW: Persona Selection
        persona_choice = input("👤 Select output persona (D for Doctor/P for Patient): ").strip().upper()
        persona = 'DOCTOR' if persona_choice == 'D' else 'PATIENT'

        if user_query and file_path:
            # We now pass the persona and use the higher top_k for robustness
            run_rag_query(user_query, file_path, persona, top_k=20) 
        else:
            print("Query and file path cannot be empty.")
    except Exception as e:
        print(f"\n🛑 A critical error occurred: {e}")