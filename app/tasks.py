import re
from pathlib import Path
from fastapi import FastAPI
from app.youtube_handling import download_subtitles, extract_full_text_from_youtube
from app.pdf_handling import extract_full_text_from_pdf
from app.training import create_embeddings
from app.llm_based_chunking import perform_ai_driven_chunking

from redis import Redis
from rq import Queue

# from app.worker_tasks import process_documents

redis_conn = Redis()
task_queue = Queue("documents", connection=redis_conn)

app = FastAPI()

async def process_file_and_embed(file_input, file_type: str, filename: str = None):
    """
    General function to process any file type, chunk it using LLM, and create embeddings.
    
    Args:
        file_input: Path to file or URL
        file_type: 'youtube' or 'pdf'
        filename: Optional filename for metadata
    """
    full_text = ""
    
    match file_type:
        case "youtube":
            # call function
            if not filename:
                video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([\w-]+)', file_input)
                filename = video_id_match.group(1) if video_id_match else "unknown_video"
            
            vtt_file_path = download_subtitles(file_input)
            full_text = extract_full_text_from_youtube(vtt_file_path)

        case "pdf":
            # call function
            if not filename:
                filename = Path(file_input).stem

            full_text = await extract_full_text_from_pdf(Path(file_input))

        case _:
            print("Unsupported file type")

    # 2. Perform AI-Driven Chunking
    chunked_documents = perform_ai_driven_chunking(full_text)
    
    # 3. Create Embeddings
    create_embeddings(chunked_documents, filename=filename, source=file_type)
    
    return filename