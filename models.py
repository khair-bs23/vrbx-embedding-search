from pydantic import BaseModel
from typing import Optional

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5 