from pydantic import BaseModel
from typing import Optional

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    collection_type: str = 'employee'  # 'employee' or 'markdown' 