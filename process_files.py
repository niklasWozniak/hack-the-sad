import os
from pathlib import Path
from typing import List, Optional, Dict, Any  # Updated this line
from sentence_transformers import SentenceTransformer
from chunker import DocumentChunker, DocumentChunk
#from embedder import model as nl_model, collection as nl_collection
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from chromadb.config import Settings


#natural language model: 
nl_model = SentenceTransformer('all-MiniLM-L6-v2')


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata to ensure all values are JSON-serializable"""
    cleaned = {}
    for key, value in metadata.items():
        # Convert None to empty string
        if value is None:
            cleaned[key] = ""
        # Convert Path objects to strings
        elif isinstance(value, Path):
            cleaned[key] = str(value)
        # Keep basic types as is
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        # Convert other types to string
        else:
            cleaned[key] = str(value)
    return cleaned

def get_simple_metadata(chunk: DocumentChunk) -> Dict[str, str]:
    """Create a simplified metadata structure with only essential fields"""
    return {
        "file_name": Path(chunk.file_path).name,
        "file_type": chunk.metadata.get("file_type", "unknown"),
        "chunk_index": str(chunk.chunk_index),
        "chunk_type": chunk.metadata.get("chunk_type", "unknown")
    }

class FileProcessor:
    def __init__(self, 
                 input_dir: str,
                 code_collection_name: str = "code_embeddings",
                 nl_collection_name: str = "natural_language_embeddings"):
        self.input_dir = Path(input_dir)
        self.chunker = DocumentChunker()
        
        # Initialize ChromaDB client
        self.client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(allow_reset=True)
        )

        #collection
        
        # Get or create collections
        self.nl_collection = self.client.get_or_create_collection(
            name=nl_collection_name,
            metadata={"description": "Natural language text embeddings"}
        )
        
        self.code_collection = self.client.get_or_create_collection(
            name=code_collection_name,
            metadata={"description": "Code embeddings using CodeBERT"}
        )
        
        # Initialize CodeBERT
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
        
    def process_directory(self, extensions: Optional[List[str]] = None):
        """Process all files in the directory with given extensions"""
        if extensions:
            extensions = {ext.lower() for ext in extensions}
        
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file():
                if extensions and file_path.suffix.lower() not in extensions:
                    continue
                    
                print(f"Processing {file_path}...")
                chunks = self.chunker.chunk_file(str(file_path))
                
                if not chunks:
                    print(f"No chunks generated for {file_path}")
                    continue
                
                # Process chunks based on file type
                file_type = self.chunker.get_file_type(str(file_path))
                if file_type == 'code':
                    self._process_code_chunks(chunks)
                else:
                    self._process_nl_chunks(chunks)
    
    def _process_nl_chunks(self, chunks: List[DocumentChunk]):
        """Process chunks using the natural language model"""
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [get_simple_metadata(chunk) for chunk in chunks]        
        # Generate embeddings
        embeddings = nl_model.encode(texts).tolist()
        
        # Store in ChromaDB
        self.nl_collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=chunk_ids,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} NL chunks to ChromaDB")
    
    def _process_code_chunks(self, chunks: List[DocumentChunk]):
        """Process chunks using CodeBERT"""
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [get_simple_metadata(chunk) for chunk in chunks]
        
        # Generate embeddings using CodeBERT
        embeddings = []
        for text in texts:
            inputs = self.code_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.code_model(**inputs)
            # Use mean pooling of the last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
            embeddings.append(embedding)
        
        # Store in ChromaDB
        self.code_collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=chunk_ids,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} code chunks to ChromaDB")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process files for vector storage")
    parser.add_argument("input_dir", help="Directory containing files to process")
    parser.add_argument("--extensions", nargs="+", help="File extensions to process (e.g., .py .js .md)")
    args = parser.parse_args()
    
    processor = FileProcessor(args.input_dir)
    processor.process_directory(args.extensions)

if __name__ == "__main__":
    main() 