# Employee Search API

A FastAPI-based application that provides semantic search capabilities for employee data using FAISS vector store and OpenAI embeddings.

## Features

- Upload Excel files containing employee data
- Create FAISS index from Excel data
- Semantic search through employee information
- RESTful API endpoints

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Upload Excel File
- **Endpoint**: `POST /upload-excel`
- **Description**: Upload an Excel file containing employee data
- **Input**: Excel file (.xlsx or .xls)
- **Response**: Success message

### 2. Search
- **Endpoint**: `GET /search`
- **Description**: Search through employee data
- **Parameters**:
  - `query`: Search query string
  - `k`: Number of results to return (default: 5)
- **Response**: List of matching results with scores

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc` 