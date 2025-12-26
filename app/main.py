from pydantic import BaseModel
from enum import Enum
import re
from typing import Optional
from fastapi import Form
from pathlib import Path
from fastapi.responses import StreamingResponse
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from app.youtube_handling import download_subtitles, extract_transcript_with_timestamps, extract_full_text_from_youtube
from app.pdf_handling import extract_full_text_from_pdf
from app.training import create_embeddings, find_top_n_similar_embeddings, generate_response_stream
from app.llm_based_chunking import perform_ai_driven_chunking

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

    
    # 1. Extract Full Text based on type
    # if file_type == 'youtube':
    #     # For YouTube, file_input is the URL
    #     if not filename:
    #         video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([\w-]+)', file_input)
    #         filename = video_id_match.group(1) if video_id_match else "unknown_video"
            
    #     vtt_file_path = download_subtitles(file_input)
    #     full_text = extract_full_text_from_youtube(vtt_file_path)
        
    # elif file_type == 'pdf':
    #     # For PDF, file_input is the file path
    #     if not filename:
    #         filename = Path(file_input).stem
            
    #     full_text = await extract_full_text_from_pdf(Path(file_input))
        
    # else:
    #     raise ValueError(f"Unsupported file type: {file_type}")
    
    # if not full_text:
    #     raise ValueError("Could not extract text from input")

    # 2. Perform AI-Driven Chunking
    chunked_documents = perform_ai_driven_chunking(full_text)
    
    # 3. Create Embeddings
    create_embeddings(chunked_documents, filename=filename, source=file_type)
    
    return filename

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EResultCode(int, Enum):
    SUCCESS = 0
    FAILURE = 1


class DataResponse(BaseModel):
    returnCode: EResultCode
    description: str


class APIResponse(BaseModel):
    dataResponse: DataResponse


class UploadURLRequest(BaseModel):
    url: str


@app.get("/")
async def read_root():
    return "Welcome to the YouTube Embeddings API!"


@app.post("/uploadURL", response_model=APIResponse)
async def upload_url(request: UploadURLRequest):
    youtube_url = request.url

    try:
        # Use the general processing function
        filename = await process_file_and_embed(youtube_url, file_type='youtube')

        return APIResponse(
            dataResponse=DataResponse(
                returnCode=EResultCode.SUCCESS,
                description=f"Documents uploaded successfully. Embeddings saved to database for video: {filename}",
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                dataResponse=DataResponse(
                    returnCode=EResultCode.FAILURE,
                    description=f"Error processing YouTube URL: {str(e)}",
                )
            ).model_dump()
        )


@app.post("/uploadPDF", response_model=APIResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=APIResponse(
                dataResponse=DataResponse(
                    returnCode=EResultCode.FAILURE,
                    description="Only PDF files are allowed.",
                )
            ).model_dump()
        )

    upload_dir = Path("static/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use the general processing function
        filename = await process_file_and_embed(file_path, file_type='pdf', filename=file.filename)

        return APIResponse(
            dataResponse=DataResponse(
                returnCode=EResultCode.SUCCESS,
                description=f"PDF processed and embeddings saved to database for file: {file.filename}",
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                dataResponse=DataResponse(
                    returnCode=EResultCode.FAILURE,
                    description=f"Error processing PDF: {str(e)}",
                )   
            ).model_dump()
        )
    finally:
        if file_path.exists():
            file_path.unlink()  # Delete the uploaded file after processing


@app.post("/generateAnswer")
async def generate_answer(
    userQuery: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    query_text = userQuery
    file_path = None  # Initialize file_path

    try:
        if file:
            allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".pdf"]
            file_extension = Path(file.filename).suffix.lower()

            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=APIResponse(
                        dataResponse=DataResponse(
                            returnCode=EResultCode.FAILURE,
                            description=f"Only image files ({', '.join(allowed_extensions[:-1])}) and PDF files are allowed.",
                        )
                    ).model_dump()
                )

            upload_dir = Path("static/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / file.filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            pdf_data = await extract_full_text_from_pdf(file_path)
            extracted_text = pdf_data
            
            print(
                f"DEBUG: Extracted text from file: {extracted_text[:200]}...")
            query_text = extracted_text  # Use extracted text as the query

        # Perform similarity search using PGVector
        # query_embedding = model.encode(userQuery).tolist()
        
        top_n_similar = find_top_n_similar_embeddings(query_text, n=3)

        if not top_n_similar:
            raise HTTPException(
                status_code=404,
                detail=APIResponse(
                    dataResponse=DataResponse(
                        returnCode=EResultCode.FAILURE,
                        description="Could not find similar embeddings. Please upload documents first.",
                    )
                ).model_dump()
            )

        return StreamingResponse(
            generate_response_stream(top_n_similar, query_text),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        error_detail = f"Error processing query: {str(e)}"
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                dataResponse=DataResponse(
                    returnCode=EResultCode.FAILURE,
                    description=error_detail,
                )
            ).model_dump()
        )
    finally:
        if file_path and file_path.exists():
            file_path.unlink()  # Delete the uploaded file after processing


