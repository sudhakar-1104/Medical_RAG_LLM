# api_server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from uvicorn import run as uvicorn_run
from typing import List

# Ensure project root is in the path to import src.query_data
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from src.query_data import run_rag_query 

# Initialize FastAPI application
app = FastAPI(title="Medical RAG Analysis API")

# --- CORS Configuration ---
# Allows the frontend (running on a different port/origin) to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Data Model ---
class QueryRequest(BaseModel):
    user_query: str
    file_path: str
    persona: str

# --- API Endpoint ---
@app.post("/analyze")
def analyze_medical_data(request: QueryRequest):
    """
    Endpoint to receive user query, target file, and persona, and run the RAG pipeline.
    """
    try:
        # Call the core RAG function
        final_report, nodes = run_rag_query(
            request.user_query, 
            request.file_path, 
            request.persona,
            top_k=20 
        )
        
        # Format nodes for JSON response
        sources = [
            {
                "file": n.metadata.get('source'), 
                "score": f"{n.score:.4f}", 
                "type": n.metadata.get('type'), 
                "content": n.get_content()
            }
            for n in nodes
        ]
        
        return {
            "report": final_report,
            "sources": sources
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"RAG EXCEPTION: {e}")
        raise HTTPException(status_code=500, detail=f"Internal RAG process failed: {e}")

if __name__ == "__main__":
    uvicorn_run(app, host="127.0.0.1", port=8000, reload=True)