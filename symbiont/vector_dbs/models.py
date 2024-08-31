from pydantic import BaseModel
from typing import Optional, List

from pydantic import Field


class ConfigsModel(BaseModel):
    vector_store: str
    vector_store_url: str
    vector_store_port: str
    vector_store_dimension: str = Field(default="1536")
    vector_store_distance: str
    vector_store_token: Optional[str] = None
    embeddings_model: str = Field(default="bge-base-en")


class VectorSearchResult(BaseModel):
    id: str
    score: float


class Metadata(BaseModel):
    source: str
    page: Optional[str]


class Document(BaseModel):
    page_content: str
    metadata: Metadata


class PineconeRecord(BaseModel):
    id: str
    values: List[float]
    metadata: Metadata


class PineconeResult(BaseModel):
    id: str
    score: float
    values: List


class PineconeResults(BaseModel):
    matches: List[PineconeResult]


class VectorRef(BaseModel):
    source: str
    page: str
    text: str


class VectorMetadata(BaseModel):
    source: str
    page: str
    text: str
