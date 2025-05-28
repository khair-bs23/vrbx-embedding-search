import os
import pandas as pd
from typing import List, Dict, Any
from langchain_community.document_loaders import DataFrameLoader, UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY
from qdrant_client import QdrantClient
from qdrant_client.http import models
import nltk
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.employee_collection = 'employee_index'
        self.markdown_collection = 'markdown_index'
        # Initialize Qdrant client
        self.client = QdrantClient(host="localhost", port=6333)
        self.employee_vectorstore = None
        self.markdown_vectorstore = None
        self._init_collections()
        # Precompile regex patterns
        self.word_pattern = re.compile(r'\b\w+\b')
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _init_collections(self):
        """Initialize or get existing collections"""
        try:
            # Get vector size from embeddings
            vector_size = len(self.embeddings.embed_query("test"))
            
            # Create collections if they don't exist
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            for collection_name in [self.employee_collection, self.markdown_collection]:
                if collection_name not in collection_names:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
            
            # Initialize vectorstores
            self.employee_vectorstore = Qdrant(
                client=self.client,
                collection_name=self.employee_collection,
                embeddings=self.embeddings
            )
            
            self.markdown_vectorstore = Qdrant(
                client=self.client,
                collection_name=self.markdown_collection,
                embeddings=self.embeddings
            )
        except Exception as e:
            print(f"Error initializing collections: {str(e)}")
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
            self.employee_vectorstore.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False

    def create_index_from_markdown(self, file_path: str) -> bool:
        """Create Qdrant index from Markdown file"""
        try:
            # Ensure collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.markdown_collection not in collection_names:
                vector_size = len(self.embeddings.embed_query("test"))
                self.client.create_collection(
                    collection_name=self.markdown_collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                # Reinitialize markdown vectorstore
                self.markdown_vectorstore = Qdrant(
                    client=self.client,
                    collection_name=self.markdown_collection,
                    embeddings=self.embeddings
                )

            # Load markdown file with enhanced parsing
            loader = UnstructuredMarkdownLoader(
                file_path,
                mode="single",  # Process as a single document
                strategy="fast"  # Use fast strategy for better performance
            )
            documents = loader.load()
            
            # Calculate dynamic chunk size based on document size
            doc_size = len(documents[0].page_content)
            target_chunks = max(10, min(30, doc_size // 1000))  # 1 chunk per 1000 chars, between 10-30 chunks
            chunk_size = max(1000, min(2000, doc_size // target_chunks))  # Between 1000 and 2000 chars
            chunk_overlap = max(200, chunk_size // 4)  # 25% overlap, minimum 200 chars
            
            # Initialize text splitter with improved parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=[
                    "\n## ",  # Split on markdown headers
                    "\n\n",   # Split on paragraphs
                    ". ",     # Split on sentences
                    "! ",     # Split on sentences
                    "? ",     # Split on sentences
                    "\n",     # Split on newlines
                    "; ",     # Split on semicolons
                    ", ",     # Split on commas
                    " ",      # Split on spaces
                    ""        # Split on characters
                ]
            )
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Filter out chunks that are too small
            chunks = [chunk for chunk in chunks if len(chunk.page_content.split()) >= 10]
            
            # Enrich chunks with metadata
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                # Get the original metadata
                metadata = chunk.metadata.copy()
                
                # Add additional metadata
                metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "source_file": os.path.basename(file_path),
                    "chunk_type": "markdown",
                    "chunk_size": len(chunk.page_content),
                    "original_doc_size": doc_size,
                    "chunk_size_ratio": len(chunk.page_content) / doc_size,
                    "created_at": datetime.now().isoformat()
                })
                
                # Create new document with enriched metadata
                enriched_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                )
                enriched_chunks.append(enriched_chunk)
            
            # Add documents to Qdrant
            self.markdown_vectorstore.add_documents(enriched_chunks)
            return True
        except Exception as e:
            print(f"Error creating markdown index: {str(e)}")
            return False

    def load_index(self, collection_type: str = 'employee') -> bool:
        """Load existing Qdrant index"""
        try:
            collection_name = self.employee_collection if collection_type == 'employee' else self.markdown_collection
            vectorstore = self.employee_vectorstore if collection_type == 'employee' else self.markdown_vectorstore
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name in collection_names:
                if collection_type == 'employee':
                    self.employee_vectorstore = Qdrant(
                        client=self.client,
                        collection_name=collection_name,
                        embeddings=self.embeddings
                    )
                else:
                    self.markdown_vectorstore = Qdrant(
                        client=self.client,
                        collection_name=collection_name,
                        embeddings=self.embeddings
                    )
                return True
            return False
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False

    def _process_result(self, doc_score_tuple: tuple, query_terms: set, collection_type: str) -> Dict[str, Any]:
        """Process a single search result with enhanced scoring"""
        doc, score = doc_score_tuple
        if score <= 0.7:  # Quick relevance check
            return None
            
        content = doc.page_content.lower()
        metadata = doc.metadata
        
        # 1. Base semantic similarity score (0-1)
        base_score = float(score)
        
        # 2. Keyword matching score
        content_words = set(self.word_pattern.findall(content))
        keyword_matches = len(query_terms.intersection(content_words))
        keyword_score = min(1.0, keyword_matches / len(query_terms)) if query_terms else 0
        
        # 3. Position score (boost chunks from the beginning of documents)
        position_score = 1.0
        if 'chunk_id' in metadata and 'total_chunks' in metadata:
            position = metadata['chunk_id'] / metadata['total_chunks']
            position_score = 1.0 - (position * 0.5)  # Reduce score by up to 50% based on position
        
        # 4. Content quality score
        content_quality = 1.0
        if len(content.split()) < 10:  # Penalize very short chunks
            content_quality = 0.7
        elif len(content.split()) > 200:  # Penalize very long chunks
            content_quality = 0.8
        
        # Combine scores with balanced weights
        final_score = (
            base_score * 0.4 +           # 50% weight to semantic similarity
            keyword_score * 0.4 +        # 30% weight to keyword matches
            position_score * 0.1 +       # 10% weight to position
            content_quality * 0.1        # 10% weight to content quality
        )
        
        # Apply minimum threshold for final results
        if final_score < 0.6:
            return None
            
        return {
            "content": doc.page_content,
            "score": final_score,
            "metadata": metadata,
            "keyword_matches": keyword_matches,
            "score_breakdown": {
                "base_score": base_score,
                "keyword_score": keyword_score,
                "position_score": position_score,
                "content_quality": content_quality
            }
        }

    def search(self, query: str, k: int = 5, collection_type: str = 'employee') -> List[Dict[str, Any]]:
        """Search the vector store using semantic search"""
        vectorstore = self.employee_vectorstore if collection_type == 'employee' else self.markdown_vectorstore
        
        if not vectorstore:
            if not self.load_index(collection_type):
                raise Exception(f"No {collection_type} index available. Please create index first.")
        
        try:
            # Get semantic search results
            results = vectorstore.similarity_search_with_score(
                query,
                k=k
            )
            
            # Return results with their scores
            return [
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                }
                for doc, score in results
            ]
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            raise

    def __del__(self):
        """Cleanup thread pool on object destruction"""
        self.executor.shutdown(wait=False)
    