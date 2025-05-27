from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
from database import VectorStore
# from pyngrok import ngrok # Remove ngrok import
import uvicorn
import logging
from datetime import datetime
from models import SearchRequest # Import the new model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Employee Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
vector_store = VectorStore()

@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    """Upload Excel file and create FAISS index"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed")
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create index from Excel file
        success = vector_store.create_index_from_excel(temp_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create index")
        
        logger.info(f"Successfully processed Excel file: {file.filename}")
        return {"message": "Excel file processed and index created successfully"}
    except Exception as e:
        logger.error(f"Error processing Excel file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search")
async def search(request: SearchRequest) -> List[Dict[str, Any]]:
    """Search the vector store"""
    query = request.query
    k = request.k
    try:
        # Log the search query
        logger.info(f"Search query received: '{query}' with k={k}")
        
        # Get search results
        results = vector_store.search(query, k=k)
        
        # Log the number of results
        logger.info(f"Found {len(results)} results for query: '{query}'")
        
        # Log each result with its score
        for idx, result in enumerate(results, 1):
            logger.info(f"Result {idx}: Score={result['score']:.4f}, Content={result['content'][:100]}...")
        
        return results
    except Exception as e:
        logger.error(f"Error processing search query '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # # Open a ngrok tunnel to the HTTP server # Remove ngrok connection code
    # public_url = ngrok.connect(8000).public_url # Remove ngrok connection code
    # logger.info(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:8000") # Remove ngrok connection code
    
    # Start the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000) 