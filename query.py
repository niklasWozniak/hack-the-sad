import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import argparse

class RAGQuery:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.HttpClient(
            host='localhost',
            port=8000,
            settings=Settings(allow_reset=True)
        )
        
        # Get collection
        self.nl_collection = self.client.get_collection('natural_language_embeddings')
        
        # Initialize model
        self.nl_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_embedding(self, query: str) -> List[float]:
        """Get embedding for a query using sentence transformer"""
        return self.nl_model.encode([query])[0].tolist()
    
    def query(self, 
             query: str, 
             n_results: int = 3, 
             where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query the natural language collection and return results
        
        Args:
            query: The search query
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {'file_type': 'text'})
        """
        results = []
        
        try:
            nl_embedding = self.get_embedding(query)
            nl_results = self.nl_collection.query(
                query_embeddings=[nl_embedding],
                n_results=n_results,
                where=where
            )
            
            # Format results
            for i in range(len(nl_results['documents'][0])):
                results.append({
                    'content': nl_results['documents'][0][i],
                    'metadata': nl_results['metadatas'][0][i],
                    'distance': nl_results['distances'][0][i] if 'distances' in nl_results else None
                })
        except Exception as e:
            print(f"Error querying collection: {e}")
        
        return results

def print_results(results: List[Dict[str, Any]]):
    """Pretty print the search results"""
    if not results:
        print("No results found.")
        return
        
    print(f'\n=== Found {len(results)} Results ===')
    for i, result in enumerate(results, 1):
        try:
            metadata = result.get('metadata', {})
            print(f'\n{i}. From: {metadata.get("file_name", "Unknown file")}')
            print(f'   Type: {metadata.get("chunk_type", "Unknown type")}')
            if result.get('distance') is not None:
                print(f'   Relevance: {1 - result["distance"]:.2%}')
            print(f'   Content: {result.get("content", "")[:200]}...')
        except Exception as e:
            print(f'\n{i}. Error displaying result: {e}')

def clear_collection():
    """Clear the natural language collection in ChromaDB"""
    try:
        client = chromadb.HttpClient(
            host='localhost',
            port=8000,
            settings=Settings(allow_reset=True)
        )
        
        nl_collection = client.get_collection(name="natural_language_embeddings")
        all_nl_items = nl_collection.get()
        all_nl_ids = all_nl_items['ids']
        
        if all_nl_ids:
            nl_collection.delete(ids=all_nl_ids)
            print(f"Successfully cleared {len(all_nl_ids)} items from natural language collection")
        else:
            print("Collection is already empty")
    except Exception as e:
        print(f"Error clearing collection: {e}")

def main():
    parser = argparse.ArgumentParser(description='Query the RAG system (Natural Language only)')
    parser.add_argument('query', nargs='?', help='The search query')
    parser.add_argument('--n', type=int, default=3, help='Number of results to return')
    parser.add_argument('--file-type', help='Filter by file type')
    parser.add_argument('--clear', action='store_true', help='Clear the natural language collection')
    args = parser.parse_args()

    if args.clear:
        clear_collection()
        return
    
    if not args.query:
        print("Error: Please provide a search query or use --clear to clear the collection")
        return
    
    # Set up search parameters
    where = {'file_type': args.file_type} if args.file_type else None
    
    # Initialize and run query
    rag = RAGQuery()
    results = rag.query(
        query=args.query,
        n_results=args.n,
        where=where
    )
    
    print_results(results)

if __name__ == '__main__':
    main()