import os
import pandas as pd
from typing import List, Dict, Any
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import FAISS_INDEX_PATH, OPENAI_API_KEY

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.index_name = 'employee_index'
        self.index_path = os.path.join(FAISS_INDEX_PATH, self.index_name)
        self.vectorstore = None

    def create_index_from_excel(self, file_path: str) -> bool:
        """Create FAISS index from Excel file"""
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
            
            # Create and save FAISS index
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.vectorstore.save_local(self.index_path)
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False

    def load_index(self) -> bool:
        """Load existing FAISS index"""
        try:
            if os.path.exists(self.index_path):
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
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
    