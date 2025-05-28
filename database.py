import os
import pandas as pd
from typing import List, Dict, Any
from langchain_community.document_loaders import DataFrameLoader, UnstructuredMarkdownLoader, WebBaseLoader
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
from bs4 import BeautifulSoup
import requests

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.employee_collection = 'employee_index'
        self.markdown_collection = 'markdown_index'
        self.webpage_collection = 'webpage_index'  # New collection for web content
        # Initialize Qdrant client
        self.client = QdrantClient(host="localhost", port=6333)
        self.employee_vectorstore = None
        self.markdown_vectorstore = None
        self.webpage_vectorstore = None
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
            
            # Initialize vectorstores dictionary
            self.vectorstores = {}
            
            # Define all collections
            all_collections = {
                self.employee_collection: "Employee data collection",
                self.markdown_collection: "Markdown documents collection",
                self.webpage_collection: "Webpage content collection"
            }
            
            # Create or get collections
            for collection_name, description in all_collections.items():
                if collection_name not in collection_names:
                    print(f"Creating collection: {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                
                # Initialize vectorstore for this collection
                self.vectorstores[collection_name] = Qdrant(
                    client=self.client,
                    collection_name=collection_name,
                    embeddings=self.embeddings
                )
            
            # Set specific vectorstore references for backward compatibility
            self.employee_vectorstore = self.vectorstores[self.employee_collection]
            self.markdown_vectorstore = self.vectorstores[self.markdown_collection]
            self.webpage_vectorstore = self.vectorstores[self.webpage_collection]
            
        except Exception as e:
            print(f"Error initializing collections: {str(e)}")
            raise

    def create_index_from_excel(self, file_path: str) -> bool:
        """Create Qdrant index from Excel file"""
        try:
            # Ensure collection exists
            if self.employee_collection not in self.vectorstores:
                self._init_collections()
            
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
            self.vectorstores[self.employee_collection].add_documents(documents)
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False

    def create_index_from_markdown(self, file_path: str) -> bool:
        """Create Qdrant index from Markdown file"""
        try:
            # Ensure collection exists
            if self.markdown_collection not in self.vectorstores:
                self._init_collections()
            
            # Load markdown file with enhanced parsing
            loader = UnstructuredMarkdownLoader(
                file_path,
                mode="elements",  # Process as separate elements
                strategy="fast"  # Use fast strategy for better performance
            )
            documents = loader.load()
            
            # Initialize text splitter with improved parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Fixed chunk size
                chunk_overlap=200,  # Fixed overlap
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
            
            # Process each document element
            all_chunks = []
            for doc in documents:
                # Skip empty documents
                if not doc.page_content.strip():
                    continue
                    
                # Split into chunks
                chunks = text_splitter.split_documents([doc])
                
                # Filter out chunks that are too small
                chunks = [chunk for chunk in chunks if len(chunk.page_content.split()) >= 10]
                
                all_chunks.extend(chunks)
            
            # Enrich chunks with metadata
            enriched_chunks = []
            for i, chunk in enumerate(all_chunks):
                # Get the original metadata
                metadata = chunk.metadata.copy()
                
                # Add additional metadata
                metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(all_chunks),
                    "source_file": os.path.basename(file_path),
                    "chunk_type": "markdown",
                    "chunk_size": len(chunk.page_content),
                    "created_at": datetime.now().isoformat()
                })
                
                # Create new document with enriched metadata
                enriched_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                )
                enriched_chunks.append(enriched_chunk)
            
            # Add documents to Qdrant
            self.vectorstores[self.markdown_collection].add_documents(enriched_chunks)
            return True
        except Exception as e:
            print(f"Error creating markdown index: {str(e)}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        # text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove empty lines
        text = '\n'.join(line for line in text.split('\n') if line.strip())
        
        return text

    def _extract_structured_content(self, text: str) -> str:
        """Extract and structure meaningful content from text"""
        # Split into lines and process each line
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip lines with only special characters or numbers
            if not re.search(r'[a-zA-Z]', line):
                continue
                
            # Skip lines that are just dates or numbers
            if re.match(r'^\d+$', line.strip()) or re.match(r'^\d{2}\s+[A-Za-z]{3},\s+\d{4}$', line.strip()):
                continue
                
            # Skip lines that are just "Apply Now" or similar
            if re.match(r'^(Apply Now|No of vacancies|:)$', line.strip()):
                continue
                
            # Clean the line
            cleaned_line = self._clean_text(line)
            if cleaned_line:
                processed_lines.append(cleaned_line)
        
        return '\n'.join(processed_lines)

    def create_index_from_webpage(self, url: str) -> bool:
        """Create or update Qdrant index from webpage content"""
        try:
            # Check if collection exists, if not create it
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.webpage_collection not in collection_names:
                print(f"Creating collection: {self.webpage_collection}")
                vector_size = len(self.embeddings.embed_query("test"))
                self.client.create_collection(
                    collection_name=self.webpage_collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                # Initialize vectorstore for the new collection
                self.vectorstores[self.webpage_collection] = Qdrant(
                    client=self.client,
                    collection_name=self.webpage_collection,
                    embeddings=self.embeddings
                )

            # First, delete ALL existing documents for this URL using metadata
            try:
                # Get collection info to check current points
                collection_info = self.client.get_collection(self.webpage_collection)
                print(f"Current points in collection: {collection_info.points_count}")
                
                # Delete using metadata filter
                self.client.delete(
                    collection_name=self.webpage_collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.source_url",
                                    match=models.MatchValue(value=url)
                                )
                            ]
                        )
                    )
                )
                
                # Verify deletion
                collection_info = self.client.get_collection(self.webpage_collection)
                print(f"Points after deletion: {collection_info.points_count}")
            except Exception as e:
                print(f"Warning: Could not delete existing documents: {str(e)}")

            # Load webpage content
            loader = WebBaseLoader(
                url,
                verify_ssl=False  # Skip SSL verification for development
            )
            documents = loader.load()

            # Initialize text splitter with improved parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Fixed chunk size
                chunk_overlap=200,  # Fixed overlap
                length_function=len,
                separators=[
                    "\n## ",  # Split on headers
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

            # Process each document
            all_chunks = []
            for doc in documents:
                # Skip empty documents
                if not doc.page_content.strip():
                    continue

                # Clean and structure the content
                cleaned_content = self._clean_text(doc.page_content)
                structured_content = self._extract_structured_content(cleaned_content)
                
                if not structured_content.strip():
                    continue

                # Create a new document with cleaned content
                cleaned_doc = Document(
                    page_content=structured_content,
                    metadata=doc.metadata
                )

                # Split into chunks
                chunks = text_splitter.split_documents([cleaned_doc])

                # Filter out chunks that are too small or contain only special characters
                chunks = [
                    chunk for chunk in chunks 
                    if len(chunk.page_content.split()) >= 10 
                    and re.search(r'[a-zA-Z]', chunk.page_content)
                ]

                all_chunks.extend(chunks)

            # Enrich chunks with metadata
            enriched_chunks = []
            timestamp = datetime.now().isoformat()
            for i, chunk in enumerate(all_chunks):
                # Create consistent metadata structure
                metadata = {
                    "chunk_id": i,
                    "total_chunks": len(all_chunks),
                    "source_url": url,
                    "chunk_type": "webpage",
                    "chunk_size": len(chunk.page_content),
                    "updated_at": timestamp,
                    "original_metadata": chunk.metadata  # Preserve original metadata
                }

                # Create new document with enriched metadata
                enriched_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                )
                enriched_chunks.append(enriched_chunk)

            # Add new documents to Qdrant
            if enriched_chunks:
                # Add documents with metadata already included in the Document objects
                self.vectorstores[self.webpage_collection].add_documents(enriched_chunks)
                
                # Verify final count
                collection_info = self.client.get_collection(self.webpage_collection)
                print(f"Final points in collection: {collection_info.points_count}")
                print(f"Successfully updated content for {url}")
                return True
            else:
                print(f"No valid content found for {url}")
                return False
            
        except Exception as e:
            print(f"Error updating webpage content: {str(e)}")
            return False

    def get_webpage_versions(self, url: str) -> List[Dict[str, Any]]:
        """Get version history for a webpage"""
        try:
            # Search for all versions of the webpage
            results = self.vectorstores[self.webpage_collection].similarity_search(
                f"source_url:{url}",
                k=100,  # Adjust based on expected number of versions
                filter={"source_url": url}
            )
            
            # Group by version hash
            versions = {}
            for doc in results:
                version_hash = doc.metadata.get('version_hash')
                if version_hash not in versions:
                    versions[version_hash] = {
                        'version_hash': version_hash,
                        'created_at': doc.metadata.get('created_at'),
                        'last_updated': doc.metadata.get('last_updated'),
                        'is_latest': doc.metadata.get('is_latest', False),
                        'chunk_count': 0
                    }
                versions[version_hash]['chunk_count'] += 1
            
            return list(versions.values())
            
        except Exception as e:
            print(f"Error getting webpage versions: {str(e)}")
            return []

    def search_webpage(self, query: str, url: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search webpage content"""
        try:
            # Build filter conditions
            filter_conditions = {}
            if url:
                filter_conditions['source_url'] = url

            # Perform search
            results = self.vectorstores[self.webpage_collection].similarity_search(
                query,
                k=k,
                filter=filter_conditions
            )

            # Format results
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get('score', 0)
                }
                for doc in results
            ]
            
        except Exception as e:
            print(f"Error searching webpage: {str(e)}")
            return []

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
        try:
            # Select the appropriate collection
            if collection_type == 'employee':
                vectorstore = self.vectorstores[self.employee_collection]
            elif collection_type == 'markdown':
                vectorstore = self.vectorstores[self.markdown_collection]
            elif collection_type == 'webpage':
                vectorstore = self.vectorstores[self.webpage_collection]
            else:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
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
    