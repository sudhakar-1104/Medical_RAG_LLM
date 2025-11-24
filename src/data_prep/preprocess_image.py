import os
import pytesseract
from PIL import Image
from llama_index.core.schema import Document
from google import genai
from dotenv import load_dotenv
import hashlib 
import uuid 

# --- CONFIGURATION & SETUP ---
load_dotenv(dotenv_path='./config/.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

def generate_image_caption_gemini(image_path: str) -> str:
    # (Gemini captioning logic remains unchanged)
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not set. Cannot use Gemini for captioning.")
        return ""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        img = Image.open(image_path)
        prompt = (
            "Analyze this medical image (e.g., X-ray, ECG, MRI). "
            "Provide a concise, detailed description of the findings, including any "
            "visible abnormalities, labels, or key characteristics. "
            "Focus on clinical relevance for diagnosis."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img]
        )
        return response.text
    except Exception as e:
        print(f"Error generating Gemini caption for {image_path}: {e}")
        return ""

def process_image_file(filepath: str) -> list[Document]:
    """
    Extracts text/caption from an image and bundles it into a single LlamaIndex Document.
    The ID is generated from the immutable file hash.
    """
    # 🟢 FIX: Use the hash of the immutable file content for the ID
    file_hash = hash_file_content(filepath)
    deterministic_base_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_hash))
    
    ocr_text = ""
    caption_text = ""
    try:
        ocr_text = pytesseract.image_to_string(Image.open(filepath))
        ocr_text = f"OCR Text: {ocr_text.strip()}\n"
    except Exception as e:
        # Note: Your error message "OCR failed..." is visible in the console output.
        print(f"OCR failed for {filepath}: {e}")

    caption_text = generate_image_caption_gemini(filepath)
    if caption_text:
        caption_text = f"Visual Description (Gemini): {caption_text.strip()}"

    combined_content = f"--- Image Analysis for {filepath.split(os.sep)[-1]} ---\n"
    combined_content += f"{ocr_text}\n{caption_text}"
    
    if not ocr_text.strip() and not caption_text.strip():
        print(f"Skipping {filepath}: Could not extract any useful text.")
        return []
        
    filename = filepath.split(os.sep)[-1]
    metadata = {'source': filename, 'type': 'image_analysis'}
    
    document = Document(
        text=combined_content.strip(), 
        metadata=metadata,
        id_=deterministic_base_id # 🟢 Set the deterministic ID directly
    ) 
    document.metadata['id'] = deterministic_base_id 

    return [document]