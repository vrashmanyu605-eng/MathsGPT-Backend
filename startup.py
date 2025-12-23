import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    uvicorn.run("app.main:app", host="0.0.0.0", port=5500,
                log_level="info", reload=True)
