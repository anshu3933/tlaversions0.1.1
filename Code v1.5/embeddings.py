from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from typing import List, Optional
from langchain.schema import Document
import os
import logging
from utils.logging_config import logger

logger = logging.getLogger(__name__)

def build_faiss_index(documents: List[Document], persist_directory: str, chunk_size: int = 1000) -> Optional[FAISS]:
    """Build an optimized FAISS index from documents.
    
    Args:
        documents (List[Document]): List of LangChain documents to index
        persist_directory (str): Directory to save the FAISS index
        chunk_size (int, optional): Size of text chunks. Defaults to 1000.
        
    Returns:
        Optional[FAISS]: Initialized and populated FAISS vectorstore
    """
    print(f"Building index from {len(documents)} documents")

    # Configure text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),  # 20% overlap
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    # Split documents into chunks
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")

    # Debug: Show sample chunks
    if texts:
        print("Sample chunk:", texts[0].page_content[:200])

    # Initialize embeddings
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key
    )

    # Build vectorstore
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save index
    vectorstore.save_local(persist_directory)
    print(f"Index saved to {persist_directory}")

    # Verify index
    test_results = vectorstore.similarity_search("test", k=1)
    print(f"Index verification: Retrieved {len(test_results)} results")

    return vectorstore

def load_faiss_index(persist_directory: str) -> Optional[FAISS]:
    """Load a FAISS index from disk.
    
    Args:
        persist_directory (str): Directory containing the FAISS index
        
    Returns:
        Optional[FAISS]: Loaded FAISS vectorstore or None if loading fails
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key
        )
        
        vectorstore = FAISS.load_local(
            persist_directory, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Verify loaded index
        test_results = vectorstore.similarity_search("test", k=1)
        print(f"Successfully loaded index with {len(test_results)} test results")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

