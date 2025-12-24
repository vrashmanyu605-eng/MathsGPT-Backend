from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json
import re
import os

def perform_ai_driven_chunking(document, max_chunks=20, fallback_chunk_size=1000):
    """
    Uses an LLM to intelligently chunk content based on semantic boundaries.
    
    Args:
        document (str): The text document to process
        max_chunks (int): Maximum number of chunks to create
        fallback_chunk_size (int): Chunk size to use if LLM chunking fails
        
    Returns:
        list: The semantically chunked documents with metadata
    """
    print("AI chunking started...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    # Initialize the Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1,
        # max_output_tokens=8192  # Increased to handle longer outputs
    )
    
    # Create a chat prompt template for the chunking task
    chunking_prompt = ChatPromptTemplate.from_template("""
    You are a document processing expert. Your task is to break down the following document into 
    at most {max_chunks} meaningful chunks. Follow these guidelines:
    
    1. Each chunk should contain complete ideas or concepts
    2. More complex sections should be in smaller chunks
    3. Preserve headers with their associated content
    4. Keep related information together
    5. Maintain the original order of the document
    6. If the document is in hindi, translate it to english before chunking.
    
    DOCUMENT:
    {document}
    
    Return ONLY a valid JSON array of strings, where each string is a chunk.
    Format your response as:
    ```json
    [
      "chunk1 text",
      "chunk2 text",
      ...
    ]
    ```
    
    Do not include any explanations or additional text outside the JSON array.
    """)
    
    # Create the chain
    chunking_chain = chunking_prompt | llm
    
    try:
        # Invoke the LLM to get chunking suggestions
        response = chunking_chain.invoke({"document": document, "max_chunks": max_chunks})
        
        # Extract JSON from the response
        content = response.content
        
        # Find JSON array in the response (looking for text between [ and ])
        json_match = re.search(r'\[\s*".*"\s*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        # Try to parse the JSON response
        chunks = json.loads(content)
        print(f"Chunked document into {len(chunks)} chunks")
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            
            # Analyze chunk complexity based on length and unique word density
            words = re.findall(r'\b\w+\b', chunk.lower())
            unique_words = set(words)
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "word_count": len(words),
                    "unique_words": len(unique_words)
                }
            )
            documents.append(doc)
        print("AI chunking ends...")
        
        print("File Writing starts....")

        file_path = Path("documents_training_data.json")

        # Load old data
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except json.JSONDecodeError:
                existing_data = []
        else:
            existing_data = []

        # Append new metadata
        existing_data.extend([doc.page_content for doc in documents])

        # Write back updated JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        return documents
            
    except Exception as e:
        print(f"LLM chunking failed: {e}")
        print("Falling back to basic chunking")
        return fallback_chunking(document, chunk_size=fallback_chunk_size)


def fallback_chunking(document, chunk_size=1000, chunk_overlap=100):
    """
    Fallback method if LLM chunking fails.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(document)
    print(f"Fallback chunking created {len(chunks)} chunks")
    
    # Convert to Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_type": "fallback",
                "document_position": round(i / len(chunks), 2)
            }
        )
        documents.append(doc)
    
    return documents
