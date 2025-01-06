from loaders import load_documents
from embeddings import build_faiss_index, load_faiss_index
from dspy_pipeline import build_faiss_index_with_dspy
from chains import build_rag_chain
import os
import shutil
from typing import Optional, Dict, Any
from utils.logging_config import logger

# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

# Define directory paths
DATA_DIR = "data"
INDEX_DIR = "models/faiss_index"

def initialize_system(data_dir: str = "data", use_dspy: bool = False) -> Optional[Dict[str, Any]]:
    """Initialize the system by loading documents and building the index."""
    try:
        # Get list of files in data directory
        file_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith(('.pdf', '.docx', '.txt'))
        ]
        
        if not file_paths:
            logger.warning(f"No documents found in {data_dir}")
            return None
            
        # Load documents (without processing)
        documents = load_documents(file_paths)
        if not documents:
            logger.error("No documents were loaded!")
            return None
            
        logger.info(f"Loaded {len(documents)} documents")
        
        # Build index (without lesson plan generation)
        vectorstore = build_faiss_index(documents, INDEX_DIR)
        if not vectorstore:
            logger.error("Failed to build vector store")
            return None
            
        # Initialize QA chain
        chain = build_rag_chain(vectorstore)
        if not chain:
            logger.error("Failed to build QA chain")
            return None
            
        return {
            "documents": documents,
            "vectorstore": vectorstore,
            "chain": chain
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return None

if __name__ == "__main__":
    # Initialize chain with configurable options
    qa_chain = initialize_system(
        data_dir=DATA_DIR,
        use_dspy=False  # Set to True to enable DSPy processing
    )
    
    if qa_chain:
        logger.info("System initialized successfully")
        
        # Optional: Add interactive testing
        while True:
            query = input("\nEnter a question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            try:
                response = qa_chain({"query": query})
                print("\nAnswer:", response['result'])
                print("\nSources:")
                for doc in response.get('source_documents', []):
                    print("-" * 40)
                    print(doc.page_content[:200] + "...")
            except Exception as e:
                logger.error(f"Error processing query: {e}")
    else:
        logger.error("Failed to initialize system")