from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
import os
import uuid 

def get_text_chunks_from_text(text: str, source_metadata: dict) -> list[Document]:
    """
    Splits raw text into smaller chunks and assigns a unique, deterministic ID 
    based on the source and the chunk's start index to enable upserting.
    """
    text_splitter = SentenceSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
    )
    
    full_doc = Document(text=text, metadata=source_metadata)
    
    # This ensures start_char_idx metadata is present
    documents = text_splitter.get_nodes_from_documents([full_doc])
    
    # 🟢 CRITICAL FIX: Generate deterministic ID using source + start index
    for doc in documents:
        # Combine source name and the chunk's starting index
        start_index = doc.metadata.get('start_char_idx', '0')
        unique_string = f"{doc.metadata.get('source')}_{start_index}"
        
        # uuid.uuid5 creates a UUID based on the hash of the unique_string
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        
        doc.id_ = doc_id
        doc.metadata['id'] = doc_id 
    
    return documents

def process_text_file(filepath: str) -> list[Document]:
    """Reads a file and returns processed documents/chunks."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = filepath.split(os.sep)[-1] 
        metadata = {'source': filename, 'type': 'text'}
        
        return get_text_chunks_from_text(content, metadata)
    except Exception as e:
        print(f"Error processing text file {filepath}: {e}")
        return []