import os
import pandas as pd
from typing import List, Dict, Any
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from config import OPENAI_API_KEY
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.collection_name = 'employee_index'
        # Initialize Qdrant client
        self.client = QdrantClient(host="localhost", port=6333)
        self.vectorstore = None
        self._init_collection()

    def _init_collection(self):
        """Initialize or get existing collection"""
        try:
            # Get vector size from embeddings
            vector_size = len(self.embeddings.embed_query("test"))
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
            
            # Initialize vectorstore
            self.vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
        except Exception as e:
            print(f"Error initializing collection: {str(e)}")
            raise

    def create_index_from_excel(self, file_path: str) -> bool:
        """Create Qdrant index from Excel file"""
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Create combined content column
            df['combined_content'] = df.apply(
                lambda row: ' | '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)]), 
                axis=1
            )
            
            # Load documents
            loader = DataFrameLoader(df, page_content_column="combined_content")
            documents = loader.load()
            
            # Add documents to Qdrant
            self.vectorstore.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False

    def load_index(self) -> bool:
        """Load existing Qdrant index"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name in collection_names:
                self.vectorstore = Qdrant(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=self.embeddings
                )
                return True
            return False
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector store"""
        if not self.vectorstore:
            if not self.load_index():
                raise Exception("No index available. Please create index first.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "score": float(score)  
            }
            for doc, score in results
        ] 
    