from google import genai
import numpy as np
import os
from typing import List, Tuple
from pathlib import Path
import google.generativeai as genai_old # Keeping for generate_content if needed, or switch fully
from dotenv import load_dotenv
import json
import psycopg2
from pgvector.psycopg2 import register_vector
from app.milvus_db import insert_embeddings_milvus, search_similar_embeddings_milvus

load_dotenv()

# Set to 'milvus' or 'pgvector'
DATABASE_BACKEND = "milvus" 

# Database connection setup
def get_db_connection():
    conn = psycopg2.connect(
        dbname="vectordb",
        user="postgres",
        password="root",
        host="localhost",
        port=5432
    )
    register_vector(conn)
    return conn

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize new Client
client = genai.Client(api_key=GEMINI_API_KEY)


genai_old.configure(api_key=GEMINI_API_KEY)
gemini_model = genai_old.GenerativeModel('gemini-2.5-pro')


def get_embedding(text: str) -> List[float]:
    try:
        res = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return res.embeddings[0].values
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise e

from langchain_core.documents import Document

def create_embeddings(input_data, filename: str = None, source: str = None):
    """
    Generates embeddings for the text and stores them in the PostgreSQL database.
    input_data: Can be a single string, a list of dictionaries, or a list of LangChain Document objects.
    """
    
    if DATABASE_BACKEND == "milvus":
        try:
             # Handle list of LangChain Documents (from LLM chunking)
            if isinstance(input_data, list) and input_data and isinstance(input_data[0], Document):
                count = 0
                for doc in input_data:
                    text_chunk = doc.page_content
                    if not text_chunk.strip():
                        continue
                        
                    print(f"DEBUG: Generating embedding for chunk {count+1} (length {len(text_chunk)})...")
                    embedding = get_embedding(text_chunk)
                    
                    # Base metadata
                    metadata = {
                        "source": source or "unknown",
                        "filename": filename or "none"
                    }
                    
                    # Merge existing metadata from Document
                    if doc.metadata:
                        metadata.update(doc.metadata)

                    insert_embeddings_milvus(filename, text_chunk, embedding, metadata)
                    count += 1
                print(f"DEBUG: Inserted {count} chunks into Milvus.")

            # Handle structured data (List of dicts)
            elif isinstance(input_data, list):
                count = 0
                for item in input_data:
                    text_chunk = item.get('text', '')
                    if not text_chunk.strip():
                        print("DEBUG: Skipping empty text chunk.")
                        continue
                        
                    print(f"DEBUG: Generating embedding for chunk {count+1} (length {len(text_chunk)})...")
                    embedding = get_embedding(text_chunk)
                    
                    # Base metadata
                    metadata = {
                        "source": source or "unknown",
                        "filename": filename or "none"
                    }
                    
                    # Add specific metadata if available
                    if 'start' in item and 'end' in item:
                        metadata['timestamp_start'] = item['start']
                        metadata['timestamp_end'] = item['end']
                    if 'page' in item:
                        metadata['page_number'] = item['page']

                    insert_embeddings_milvus(filename, text_chunk, embedding, metadata)
                    count += 1
                print(f"DEBUG: Inserted {count} chunks into Milvus.")
            
            # Handle simple string data
            elif isinstance(input_data, str):
                print("DEBUG: Generating embedding for single string...")
                embedding = get_embedding(input_data)
                metadata = {
                    "source": source or "unknown",
                    "filename": filename or "none"
                }
                insert_embeddings_milvus(filename, input_data, embedding, metadata)
                print("DEBUG: Inserted single chunk into Milvus.")
                
            print(f"Saved embeddings in Milvus for {filename}")
            return # Exit
            
        except Exception as e:
            print(f"Error saving embedding to Milvus: {e}")
            raise e

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Handle list of LangChain Documents (from LLM chunking)
            if isinstance(input_data, list) and input_data and isinstance(input_data[0], Document):
                count = 0
                for doc in input_data:
                    text_chunk = doc.page_content
                    if not text_chunk.strip():
                        continue
                        
                    print(f"DEBUG: Generating embedding for chunk {count+1} (length {len(text_chunk)})...")
                    embedding = get_embedding(text_chunk)
                    
                    # Base metadata
                    metadata = {
                        "source": source or "unknown",
                        "filename": filename or "none"
                    }
                    
                    # Merge existing metadata from Document
                    if doc.metadata:
                        metadata.update(doc.metadata)

                    cursor.execute(
                        """
                        INSERT INTO documents (filename, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (filename, text_chunk, embedding, json.dumps(metadata))
                    )
                    count += 1
                print(f"DEBUG: Inserted {count} chunks into DB.")

            # Handle structured data (List of dicts)
            elif isinstance(input_data, list):
                count = 0
                for item in input_data:
                    text_chunk = item.get('text', '')
                    if not text_chunk.strip():
                        print("DEBUG: Skipping empty text chunk.")
                        continue
                        
                    print(f"DEBUG: Generating embedding for chunk {count+1} (length {len(text_chunk)})...")
                    embedding = get_embedding(text_chunk)
                    
                    # Base metadata
                    metadata = {
                        "source": source or "unknown",
                        "filename": filename or "none"
                    }
                    
                    # Add specific metadata if available
                    if 'start' in item and 'end' in item:
                        metadata['timestamp_start'] = item['start']
                        metadata['timestamp_end'] = item['end']
                    if 'page' in item:
                        metadata['page_number'] = item['page']

                    cursor.execute(
                        """
                        INSERT INTO documents (filename, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (filename, text_chunk, embedding, json.dumps(metadata))
                    )
                    count += 1
                print(f"DEBUG: Inserted {count} chunks into DB.")
                
            # Handle simple string data
            elif isinstance(input_data, str):
                print("DEBUG: Generating embedding for single string...")
                embedding = get_embedding(input_data)
                metadata = {
                    "source": source or "unknown",
                    "filename": filename or "none"
                }
                cursor.execute(
                    """
                    INSERT INTO documents (filename, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (filename, input_data, embedding, json.dumps(metadata))
                )
                print("DEBUG: Inserted single chunk into DB.")
            
            conn.commit()
            print(f"Saved embeddings in DB for {filename}")
            
    except Exception as e:
        conn.rollback()
        print(f"Error saving embedding: {e}")
        raise e
    finally:
        conn.close()


def find_top_n_similar_embeddings(query_text: str, n: int) -> List[Tuple]:
    """
    Finds the top N most similar embeddings from the database.
    Returns a list of tuples: (filename, content, metadata, distance)
    """
    query_embedding = get_embedding(query_text)

    if DATABASE_BACKEND == "milvus":
        return search_similar_embeddings_milvus(query_embedding, top_k=n)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Cast the parameter to vector type explicitly
            cursor.execute(
                """
                SELECT filename, content, metadata
                FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (query_embedding, n)
            )
            results = cursor.fetchall()
    finally:
        conn.close()

    return results


def generate_response_stream(context_data: List[Tuple], user_query: str):
    """
    Generates a streamed response from the Gemini Pro model based on the given context and user query.
    context_data: List of tuples (filename, content, metadata)
    """
    
    # Prepare context text with source information
    formatted_context_parts = []
    for filename, content, metadata in context_data:
        source_info = f"Source: {filename}"
        
        # Parse metadata to extract details
        meta_dict = {}
        if isinstance(metadata, str):
            try:
                meta_dict = json.loads(metadata)
            except:
                pass
        elif isinstance(metadata, dict):
            meta_dict = metadata

        if 'source' in meta_dict and meta_dict['source'] != filename:
             source_info += f" ({meta_dict['source']})"
        
        # Add timestamp or page info
        if 'timestamp_start' in meta_dict and 'timestamp_end' in meta_dict:
            source_info += f" [Time: {meta_dict['timestamp_start']} - {meta_dict['timestamp_end']}]"
        if 'page_number' in meta_dict:
            source_info += f" [Page: {meta_dict['page_number']}]"
        
        formatted_context_parts.append(f"--- {source_info} ---\n{content}\n")

    context_text = "\n".join(formatted_context_parts)

    print(f"DEBUG: Context prepared with {len(context_data)} chunks.")
    # Log the detailed sources for debugging
    print(f"DEBUG: Sources used:\n{context_text}")

    latex_matrix_example = r"""
$$
\begin{bmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{bmatrix}
\begin{bmatrix}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4}
\end{bmatrix}
=
\begin{bmatrix}
b_{1} \\
b_{2} \\
b_{3} \\
b_{4}
\end{bmatrix}
$$
"""

    prompt = f"""
You are a mathematics teaching assistant.

You are given a transcript from a Gajendra Purohit YouTube lecture or a PDF document.
This transcript is the ONLY source of truth.

--------------------
TRANSCRIPT CONTEXT:
{context_text}
--------------------

USER QUESTION:
{user_query}

INSTRUCTIONS (STRICT):
- Answer ONLY the question asked by the user.
- Use information STRICTLY from the transcript context.
- Do NOT introduce new formulas, steps, or explanations not present in the transcript.
- If the exact answer is not found in the transcript, respond with:
  "The answer to this question is not explicitly available in the provided transcript."
- ALWAYS cite the source (filename) when using information from a specific chunk. E.g., "According to [filename]..."

### MATHEMATICAL FORMATTING RULES (MANDATORY):
- If a matrix or augmented matrix is present:
  - EACH ROW MUST BE ON A NEW LINE
  - Use spaces for column alignment
  - Use `|` ONLY to separate augmented columns
  - NEVER write matrices in a single line
  - Do NOT compress rows into one sentence

- Example format to FOLLOW EXACTLY:
{latex_matrix_example}

- Rewrite or format the answer clearly and concisely.
- Use proper mathematical notation.
- Do NOT add examples, intuition, commentary, or extra theory.

FINAL ANSWER:
"""

    try:
        response = gemini_model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                # print(f"DEBUG: Gemini chunk: {chunk.text}")context_data
                yield chunk.text
            else:
                pass
    except Exception as e:
        print(f"ERROR: from Gemini API: {str(e)}")
        yield f"Error from Gemini API: {str(e)}"

