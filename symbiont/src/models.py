from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class LLMModel(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4"
    LLAMA = "llama"


class FileUploadResponse(BaseModel):
    identifier: str
    file_name: str
    url: str
    download_url: str


class ReourceCategory(str, Enum):
    PDF = ("pdf",)
    VIDEO = ("video",)
    AUDIO = ("audio",)
    WEBPAGE = "webpage"


class Text(BaseModel):
    text: str


class Resource(BaseModel):
    category: ReourceCategory
    identifier: str
    name: str
    url: str


class Chat(BaseModel):
    bot: List[str] = []
    user: List[str] = []


class Study(BaseModel):
    chat: Chat
    createdAt: str
    description: str
    name: str
    resources: List[Resource]
    userId: str
    studyId: str


class TextUpdateRequest(BaseModel):
    studyId: str
    text: str


class UploadResource(Resource):
    studyId: str
    userId: str


class AddChatMessageRequest(BaseModel):
    studyId: str
    message: str
    userId: str
    role: str


class ChatMessage(BaseModel):
    role: str
    content: str
    createdAt: datetime


class StudyResource(BaseModel):
    studyId: str
    name: str
    url: str
    identifier: str
    category: str
    summary: str = ""


class ResourceUpload(BaseModel):
    studyId: str


class GetResourcesResponse(BaseModel):
    resources: List[StudyResource]


class ChatRequest(BaseModel):
    user_query: str
    previous_message: Optional[str] = None
    study_id: str
    resource_identifier: Optional[str] = None


class EmbeddingModels(str, Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


class DocumentPage(BaseModel):
    page_content: str
    metadata: dict = {"source": str, "page": 0, "text": str}
    type: str = "Document"


class PineconeRecord(BaseModel):
    id: str
    values: List[float]
    metadata: dict = {"text": str, "source": str, "pageNumber": 0}


class ProcessYoutubeVideoRequest(BaseModel):
    studyId: str
    url: str


class ProcessWebpageResourceRequest(BaseModel):
    urls: List[str]
    studyId: str
