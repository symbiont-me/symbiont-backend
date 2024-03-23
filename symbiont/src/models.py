from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class LLMModel(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_Turbo_0125 = "gpt-3.5-turbo-0125"
    GPT_3_5_Turbo_1106 = "gpt-3.5-turbo-1106"
    GPT_4_Turbo_Preview = "gpt-4-turbo-preview"
    GPT_4_1106_Preview = "gpt-4-1106-preview"
    GPT_4 = "gpt-4"
    GPT_4_32k = "gpt-4-32k"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_3_SONNET_2024022 = "claude-3-sonnet-2024022"
    CLAUDE_3_HAIKU_20240307 = "claude-3-haiku-20240307"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"
    GEMINI_PRO = "gemini-pro"
    GEMINI_1_PRO = "gemini-1.0-pro"
    GEMINI_1_PRO_001 = "gemini-1.0-pro-001"
    GEMINI_1_PRO_LATEST = "gemini-1.0-pro-latest"


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
    image: str
    resources: List[Resource]
    userId: str


class CreateStudyRequest(BaseModel):
    name: str = ""
    description: str = ""
    image: str = ""


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
    combined: Optional[bool] = False


class EmbeddingModels(str, Enum):
    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPENAI_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    OPENAI_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    EMBEDDING_001 = "embedding-001"
    VOYAGEAI_LARGE_2 = "voyage-large-2"
    VOYAGEAI_2 = "voyage-2"
    VOYAGEAI_LIGHT_02_INSTRUCT = "voyage-lite-02-instruct"


class DocumentPage(BaseModel):
    page_content: str
    metadata: dict = {"source": str, "page": 0, "text": str}
    type: str = "Document"


class PineconeRecord(BaseModel):
    id: str
    values: List[float]
    metadata: dict = {"text": str, "source": str, "pageNumber": 0}


class AddYoutubeVideoRequest(BaseModel):
    studyId: str
    url: HttpUrl


class AddWebpageResourceRequest(BaseModel):
    urls: List[HttpUrl]
    studyId: str
