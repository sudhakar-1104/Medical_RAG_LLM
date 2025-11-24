import os
import assemblyai as aai
from llama_index.core.schema import Document
from dotenv import load_dotenv
import uuid
import hashlib # 🟢 Added for File Hashing

# --- CONFIGURATION & SETUP ---
load_dotenv(dotenv_path='./config/.env') 
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")

if ASSEMBLY_API_KEY:
    aai.settings.api_key = ASSEMBLY_API_KEY

def hash_file_content(filepath: str) -> str:
    """Generates a SHA-256 hash based on the file's binary content."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error hashing file {filepath}: {e}")
        return str(uuid.uuid4())

def process_audio_file(filepath: str) -> list[Document]:
    """
    Transcribes an audio file into text and returns it as a LlamaIndex Document 
    with a deterministic ID derived from the file hash.
    """
    if not ASSEMBLY_API_KEY:
        print("AssemblyAI API key not loaded from .env. Skipping audio processing.")
        return []

    # 🟢 FIX: Generate a stable ID based on the immutable audio file content
    file_hash = hash_file_content(filepath)
    deterministic_base_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_hash))
    
    print(f"Transcribing audio file: {filepath} using AssemblyAI...")
    
    try:
        config = aai.TranscriptionConfig(language_code="en")
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(filepath)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"AssemblyAI Error for {filepath}: {transcript.error}")
            return []
            
        transcription = transcript.text.strip()
        
        if not transcription:
            print(f"Transcription failed or was empty for {filepath}.")
            return []
            
        content = f"--- AssemblyAI Transcription ---\n{transcription}"
        filename = filepath.split(os.sep)[-1]
        
        document = Document(
            text=content, 
            metadata={'source': filename, 'type': 'audio_transcription'},
            id_=deterministic_base_id # 🟢 Use the stable file hash ID
        )
        document.metadata['id'] = deterministic_base_id 

        return [document]
        
    except Exception as e:
        print(f"Error during AssemblyAI transcription for {filepath}: {e}")
        return []