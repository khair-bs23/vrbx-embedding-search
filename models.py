from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    collection_type: Optional[str] = 'employee'  # 'employee', 'markdown', or 'webpage'

class WebpageRequest(BaseModel):
    url: HttpUrl

class WebpageVersionRequest(BaseModel):
    url: HttpUrl
    version_hash: Optional[str] = None

class WebpageVersion(BaseModel):
    version_hash: str
    created_at: str
    last_updated: str
    is_latest: bool
    chunk_count: int

class WebpageSearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float 