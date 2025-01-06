from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import BaseRetriever
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

def build_rag_chain(
    vectorstore: BaseRetriever,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    max_tokens: int = 2048,
    k_documents: int = 4
) -> Optional[RetrievalQA]:
    """Build an optimized RAG chain with configurable parameters.
    
    Args:
        vectorstore (BaseRetriever): Initialized vector store for document retrieval
        model_name (str): Name of the OpenAI model to use
        temperature (float): Temperature for response generation
        max_tokens (int): Maximum tokens in response
        k_documents (int): Number of documents to retrieve
        
    Returns:
        Optional[RetrievalQA]: Initialized RAG chain or None if initialization fails
    """
    try:
        # 1. Create enhanced prompt template
        prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question. 
If you cannot answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # 2. Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )

        # 3. Configure retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k_documents,
                "fetch_k": k_documents * 2,  # Fetch more docs initially for better filtering
                "score_threshold": None  # No threshold to ensure we get results
            }
        )

        # 4. Build chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True,
                "document_separator": "\n\n"  # Clear separation between documents
            }
        )

        # 5. Test chain
        test_response = chain({"query": "test"})
        print("Chain test response:", test_response.keys())

        # Verify chain
        if chain:
            try:
                test_response = chain({"query": "test"})
                if not isinstance(test_response, dict) or "result" not in test_response:
                    logger.warning("Chain verification failed: invalid response format")
                    return None
            except Exception as e:
                logger.error(f"Chain verification failed: {e}")
                return None
            
        return chain

    except Exception as e:
        print(f"Error building RAG chain: {e}")
        return None
