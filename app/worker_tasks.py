import asyncio
from app.tasks import process_file_and_embed

async def async_wrapper(job):
    file_input = job["file_input"]
    file_type = job["file_type"]
    filename = job.get("filename")

    await process_file_and_embed(file_input, file_type, filename)

def process_documents(job):
    asyncio.run(async_wrapper(job))