# import shutil
# import google.generativeai as genai
# from fastapi import HTTPException
# import fitz  # PyMuPDF
# import gc
# import os
# import logging
# from PIL import Image
# from pathlib import Path

# logger = logging.getLogger(__name__)

# GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables")

# genai.configure(api_key=GEMINI_API_KEY)
# # Initialize the Gemini Pro model
# gemini_model = genai.GenerativeModel('gemini-2.5-pro')


# def pdf_to_images_combined(pdf_inputs, output_dir: Path) -> list[Path]:
#     """Convert PDFs to images with proper resource cleanup"""
#     if isinstance(pdf_inputs, (str, Path)):
#         pdf_files = [Path(pdf_inputs)]
#     elif isinstance(pdf_inputs, list):
#         pdf_files = [Path(p) if isinstance(p, str) else p for p in pdf_inputs]
#     else:
#         logger.error("Invalid input: Must be a Path, string, or list of Paths")
#         return []

#     all_image_paths = []
#     output_dir.mkdir(parents=True, exist_ok=True)

#     for pdf_index, pdf_file in enumerate(pdf_files, 1):
#         logger.info(
#             f"Converting PDF {pdf_index}/{len(pdf_files)}: {pdf_file.name}")
#         doc = None

#         try:
#             doc = fitz.open(pdf_file)

#             for page_index in range(len(doc)):
#                 try:
#                     global_page_num = len(all_image_paths) + 1
#                     page = doc.load_page(page_index)
#                     pix = page.get_pixmap(dpi=300)

#                     img_path = output_dir / f"page_{global_page_num:03d}.png"
#                     pix.save(str(img_path))
#                     all_image_paths.append(img_path)

#                     # Explicit cleanup
#                     del pix
#                     del page

#                 except Exception as e:
#                     logger.error(f"Error on page {page_index+1}: {e}")
#                     continue

#         except Exception as e:
#             logger.error(f"Failed to open {pdf_file}: {e}")

#         finally:
#             if doc:
#                 doc.close()
#             gc.collect()  # Force garbage collection

#     logger.info(f"✅ Converted {len(all_image_paths)} pages total")
#     return all_image_paths


# async def extract_text_with_page_numbers(file_path: Path) -> list[dict]:
#     """
#     Extracts text from a PDF, keeping track of page numbers.
#     Returns a list of dictionaries: [{'page': 1, 'text': '...'}, ...]
#     """
#     file_name = file_path.name
#     _, file_extension = os.path.splitext(file_name)
#     file_extension = file_extension.lower()

#     extracted_pages = []
#     temp_output_dir = None

#     try:
#         if not file_path.exists():
#             raise HTTPException(
#                 status_code=404, detail=f"File not found at path: {file_path}")

#         if file_extension == ".pdf":
#             temp_output_dir = Path("static/temp_pdf_images")
#             temp_output_dir.mkdir(parents=True, exist_ok=True)

#             image_paths = pdf_to_images_combined(file_path, temp_output_dir)

#             if not image_paths:
#                 raise HTTPException(
#                     status_code=500, detail="Failed to convert PDF to images.")

#             # Processing images sequentially to maintain page order
#             for i, img_path in enumerate(image_paths):
#                 try:
#                     with Image.open(img_path) as image:
#                         model = genai.GenerativeModel('gemini-2.5-flash')
#                         response = model.generate_content(["", image])
#                         extracted_img_text = response.text
                        
#                         # Page numbers are 1-based
#                         extracted_pages.append({
#                             'page': i + 1,
#                             'text': extracted_img_text
#                         })
                        
#                         print(f"DEBUG: Extracted text from page {i+1}: {extracted_img_text[:100]}...")

#                 except Exception as img_e:
#                     logger.error(f"Error processing image {img_path}: {img_e}")

#         elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
#             # Treat single image as page 1
#             with Image.open(file_path) as image:
#                 model = genai.GenerativeModel('gemini-2.5-flash')
#                 response = model.generate_content(["", image])
#                 extracted_img_text = response.text
#                 extracted_pages.append({
#                     'page': 1,
#                     'text': extracted_img_text
#                 })

#         else:
#             raise HTTPException(
#                 status_code=400, detail=f"Unsupported file type: {file_extension}")

#     except Exception as e:
#         error_detail = f"Failed to extract text: {str(e)}"
#         raise HTTPException(status_code=500, detail=error_detail)
#     finally:
#         if temp_output_dir and temp_output_dir.exists():
#             shutil.rmtree(temp_output_dir)
#             logger.info(f"Cleaned up temporary directory: {temp_output_dir}")

#     return extracted_pages

# # async def extract_text_from_document(file_path: Path):
# #     # Backward compatibility wrapper
# #     pages = await extract_text_with_page_numbers(file_path)
# #     return "\n".join([p['text'] for p in pages])

# async def extract_full_text_from_pdf(file_path: Path) -> str:
#     """
#     Extracts the full text from a PDF file as a single string.
#     This is used for LLM-based chunking which prefers the whole document context.
#     """
#     # pages = await extract_text_with_page_numbers(file_path)
#     # Combine all page texts with double newlines to separate pages
#     # full_text = "\n\n".join([p['text'] for p in pages])
#     # return full_text\

#     try:
#         client = genai.client(api_key = GEMINI_API_KEY)

#         prompt = f"""
#         Extract all the text from this file and provide it as a single continuous string without any additional commentary or formatting.
#         """

#         sample_file = client.files.upload(file=file_path)

#         response = client.models.generate_content(
#             model="gemini-2.5-flash-lite",
#             contents=[sample_file, prompt],
#         )

#         # Collect output text
#         full_text = ""

#         if response and response.candidates:
#             for part in response.candidates[0].content.parts:
#                 if hasattr(part, "text"):
#                     print(part.text)
#                     full_text += part.text  # Append to file buffer
#         else:
#             full_text = "No text response received!"
#             print(full_text)

#         # Save to output file
#         with open("output.txt", "w", encoding="utf-8") as file:
#             file.write(full_text)

#         print("\nSaved response to output.txt")
#         return full_text

#     except Exception as e:
#         print ("Error occured ", e)

import shutil
from google import genai
from fastapi import HTTPException
import fitz  # PyMuPDF
import gc
import os
import logging
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize the client with the new library
client = genai.Client(api_key=GEMINI_API_KEY)


# def pdf_to_images_combined(pdf_inputs, output_dir: Path) -> list[Path]:
#     """Convert PDFs to images with proper resource cleanup"""
#     if isinstance(pdf_inputs, (str, Path)):
#         pdf_files = [Path(pdf_inputs)]
#     elif isinstance(pdf_inputs, list):
#         pdf_files = [Path(p) if isinstance(p, str) else p for p in pdf_inputs]
#     else:
#         logger.error("Invalid input: Must be a Path, string, or list of Paths")
#         return []

#     all_image_paths = []
#     output_dir.mkdir(parents=True, exist_ok=True)

#     for pdf_index, pdf_file in enumerate(pdf_files, 1):
#         logger.info(
#             f"Converting PDF {pdf_index}/{len(pdf_files)}: {pdf_file.name}")
#         doc = None

#         try:
#             doc = fitz.open(pdf_file)

#             for page_index in range(len(doc)):
#                 try:
#                     global_page_num = len(all_image_paths) + 1
#                     page = doc.load_page(page_index)
#                     pix = page.get_pixmap(dpi=300)

#                     img_path = output_dir / f"page_{global_page_num:03d}.png"
#                     pix.save(str(img_path))
#                     all_image_paths.append(img_path)

#                     # Explicit cleanup
#                     del pix
#                     del page

#                 except Exception as e:
#                     logger.error(f"Error on page {page_index+1}: {e}")
#                     continue

#         except Exception as e:
#             logger.error(f"Failed to open {pdf_file}: {e}")

#         finally:
#             if doc:
#                 doc.close()
#             gc.collect()  # Force garbage collection

#     logger.info(f"✅ Converted {len(all_image_paths)} pages total")
#     return all_image_paths


# async def extract_text_with_page_numbers(file_path: Path) -> list[dict]:
#     """
#     Extracts text from a PDF, keeping track of page numbers.
#     Returns a list of dictionaries: [{'page': 1, 'text': '...'}, ...]
#     """
#     file_name = file_path.name
#     _, file_extension = os.path.splitext(file_name)
#     file_extension = file_extension.lower()

#     extracted_pages = []
#     temp_output_dir = None

#     try:
#         if not file_path.exists():
#             raise HTTPException(
#                 status_code=404, detail=f"File not found at path: {file_path}")

#         if file_extension == ".pdf":
#             temp_output_dir = Path("static/temp_pdf_images")
#             temp_output_dir.mkdir(parents=True, exist_ok=True)

#             image_paths = pdf_to_images_combined(file_path, temp_output_dir)

#             if not image_paths:
#                 raise HTTPException(
#                     status_code=500, detail="Failed to convert PDF to images.")

#             # Processing images sequentially to maintain page order
#             for i, img_path in enumerate(image_paths):
#                 try:
#                     with Image.open(img_path) as image:
#                         # Upload the image file
#                         uploaded_file = client.files.upload(path=str(img_path))
                        
#                         # Generate content using the new API
#                         response = client.models.generate_content(
#                             model='gemini-2.0-flash-exp',
#                             contents=[uploaded_file, "Extract all text from this image"]
#                         )
                        
#                         extracted_img_text = response.text
                        
#                         # Page numbers are 1-based
#                         extracted_pages.append({
#                             'page': i + 1,
#                             'text': extracted_img_text
#                         })
                        
#                         print(f"DEBUG: Extracted text from page {i+1}: {extracted_img_text[:100]}...")

#                 except Exception as img_e:
#                     logger.error(f"Error processing image {img_path}: {img_e}")

#         elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
#             # Treat single image as page 1
#             uploaded_file = client.files.upload(path=str(file_path))
            
#             response = client.models.generate_content(
#                 model='gemini-2.5-flash-lite',
#                 contents=[uploaded_file, "Extract all text from this image"]
#             )
            
#             extracted_img_text = response.text
#             extracted_pages.append({
#                 'page': 1,
#                 'text': extracted_img_text
#             })

#         else:
#             raise HTTPException(
#                 status_code=400, detail=f"Unsupported file type: {file_extension}")

#     except Exception as e:
#         error_detail = f"Failed to extract text: {str(e)}"
#         raise HTTPException(status_code=500, detail=error_detail)
#     finally:
#         if temp_output_dir and temp_output_dir.exists():
#             shutil.rmtree(temp_output_dir)
#             logger.info(f"Cleaned up temporary directory: {temp_output_dir}")

#     return extracted_pages


async def extract_full_text_from_pdf(file_path: Path) -> str:
    """
    Extracts the full text from a PDF file as a single string.
    This is used for LLM-based chunking which prefers the whole document context.
    """
    try:
        prompt = """
        Extract all the text from this file and provide it as a single continuous string without any additional commentary or formatting.
        """

        # Upload the file
        sample_file = client.files.upload(file=file_path)

        # Generate content
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[sample_file, prompt],
        )

        # Collect output text
        full_text = response.text if response.text else "No text response received!"
        
        print(full_text)

        # Save to output file
        # with open("output.txt", "w", encoding="utf-8") as file:
        #     file.write(full_text)

        # print("\nSaved response to output.txt")
        return full_text

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise