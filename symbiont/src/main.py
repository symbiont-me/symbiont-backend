from re import A
from fastapi import BackgroundTasks, FastAPI, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from firebase_admin import initialize_app, firestore, auth, credentials, storage
from typing import List
from pydantic.networks import HttpUrl
from enum import Enum
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import OpenAI, OpenAIEmbeddings
import os
from langchain.chains import LLMChain
from dotenv import load_dotenv
from typing import AsyncGenerator
from datetime import datetime
from google.cloud.firestore import ArrayUnion
from .utils import (
    make_file_identifier,
    verify_token,
)
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from hashlib import md5

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
pinecone_endpoint = os.getenv("PINECONE_API_ENDPOINT")


app = FastAPI()

pc = Pinecone(api_key=pinecone_api_key, endpoint=pinecone_endpoint)
index = pc.Index("symbiont-me")
# wait for index to be initialized
while not pc.describe_index("symbiont-me").status["ready"]:
    time.sleep(1)

if index is None:
    raise Exception("Pinecone index not found")


origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       FIREBASE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cred = credentials.Certificate("src/serviceAccountKey.json")
initialize_app(cred, {"storageBucket": "symbiont-e7f06.appspot.com"})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Firebase Storage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def download_from_firebase_storage(file_key: str) -> str:
    bucket = storage.bucket()
    blob = bucket.blob(file_key)
    if not os.path.exists("temp"):
        os.makedirs("temp")
    save_path = f"temp/{file_key}"
    blob.download_to_filename(save_path)
    return save_path


async def delete_local_file(file_path: str):
    os.remove(file_path)


# ~~~~~~~~~~~~~~~~~~~~
#       PINECONE
# ~~~~~~~~~~~~~~~~~~~~


class EmbeddingModels(str, Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


# TODO move to a separate file
class PdfPage(BaseModel):
    page_content: str
    metadata: dict = {"source": str, "page": 0, "text": str}
    type: str = "Document"


class PineconeRecord(BaseModel):
    id: str
    values: List[float]
    metadata: dict = {"text": str, "source": str, "pageNumber": 0}


# Initialize the OpenAIEmbeddings object, will use the same object for embedding tasks
embed = OpenAIEmbeddings(model=EmbeddingModels.TEXT_EMBEDDING_3_SMALL, dimensions=1536)


async def embed_document(doc: PdfPage) -> PineconeRecord:

    vec = await embed.aembed_query(doc.page_content)
    hash = md5(doc.page_content.encode("utf-8")).hexdigest()
    return PineconeRecord(id=hash, values=vec, metadata=doc.metadata)


async def prepare_resource_for_pinecone(file_identifier: str):
    file_path = await download_from_firebase_storage(file_identifier)
    if file_path is not None and file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = [PdfPage(**page.dict()) for page in loader.load_and_split()]

        docs = []
        for page in pages:
            prepared_pages = await prepare_pdf_for_pinecone(page)
            docs.extend(prepared_pages)
        vecs = [await embed_document(doc) for doc in docs]

        await upload_vecs_to_pinecone(vecs, file_identifier)
        await delete_local_file(file_path)


async def upload_vecs_to_pinecone(vecs: List[PineconeRecord], file_identifier: str):
    client = Pinecone(api_key=pinecone_api_key, endpoint=pinecone_endpoint)
    index = client.Index("symbiont-me")
    formatted_vecs = [(vec.id, vec.values, vec.metadata) for vec in vecs]
    if index is None:
        raise Exception("Pinecone index not found")
    index.upsert(vectors=formatted_vecs, namespace=file_identifier)
    print("Uploaded to Pinecone")


def truncate_string_by_bytes(string, num_bytes):
    encoded_string = string.encode("utf-8")
    truncated_string = encoded_string[:num_bytes]
    return truncated_string.decode("utf-8", "ignore")


async def prepare_pdf_for_pinecone(pdf_page: PdfPage) -> List[PdfPage]:
    page_content = pdf_page.page_content.replace("\n", "")
    page_content = truncate_string_by_bytes(page_content, 10000)
    # TODO use NLTK Splitter with db reference and don't store text in pinecone
    # Pincecone is for embeddings only, it is expensive to store text in pinecone
    # text_splitter = NLTKTextSplitter()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    split_texts = text_splitter.create_documents([page_content])
    docs = [
        PdfPage(
            page_content=split_text.page_content,
            metadata={
                "text": split_text.page_content,
                "source": pdf_page.metadata["source"],
                "page": pdf_page.metadata["page"],
            },
            type=pdf_page.type,
        )
        for split_text in split_texts
    ]
    return docs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Vector Search
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_query_embedding(query: str) -> List[float]:
    vec = embed.embed_query(query)
    return vec


def search_pinecone_index(query: str, file_identifier: str):
    # TODO don't initialize Pinecone client here
    client = Pinecone(api_key=pinecone_api_key, endpoint=pinecone_endpoint)
    index = client.Index("symbiont-me")

    if index is None:
        raise Exception("Pinecone index not found")

    query_embedding = get_query_embedding(query)
    query_matches = index.query(
        vector=query_embedding,
        top_k=2,
        namespace=file_identifier,
        include_metadata=True,
    )
    return query_matches


def get_chat_context(query: str, file_identifier: str):
    result = search_pinecone_index(query, file_identifier)
    context = ""
    for match in result.matches:
        context += match.metadata["text"] + " "
    # TODO return an object with matches for detailed footnoting
    return context


# test_query = "What is zcash?"
# test_namespace = "20240223213906_1902.07337v1.pdf"
# get_query_embedding(test_query)
# search_pinecone_index(test_query, test_namespace)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       MODELS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LLMModel(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    GPT_4 = "gpt-4"
    LLAMA = "llama"


class FileUploadResponse(BaseModel):
    identifier: str
    file_name: str
    url: str


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
    userId: str


class UploadResource(Resource):
    studyId: str
    userId: str


class AddChatMessageRequest(BaseModel):
    studyId: str
    message: str
    userId: str
    role: str


class ChatRequest(BaseModel):
    message: str
    studyId: str
    resource_identifier: str | None


class ChatMessage(BaseModel):
    role: str
    content: str
    createdAt: datetime


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       USER STUDIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@app.post("/get-user-studies")
async def get_user_studies(decoded_token: dict = Depends(verify_token)):
    try:
        userId = decoded_token["uid"]
        db = firestore.client()
        studies_ref = db.collection("studies_")
        query = studies_ref.where("userId", "==", userId)
        studies = query.stream()

        # Create a list of dictionaries, each containing the studyId and the study's data
        studies_data = [
            {"id": study.id, **(study.to_dict() or {})} for study in studies
        ]
        return {"studies": studies_data}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred while fetching user studies.",
                "details": str(e),
            },
        )


@app.post("/create-study/")
async def create_study(study: Study):
    db = firestore.client()
    doc_ref = db.collection("studies_").document()
    doc_ref.set(study.model_dump())

    return {"message": "Study created successfully", "study": study.model_dump()}


@app.post("/get-study/")
async def get_study(studyId: str):
    db = firestore.client()
    # TODO verify user has access to study
    study_ref = db.collection("studies_").document(studyId)
    study = study_ref.get()
    if study.exists:
        print("Document data:", study.to_dict())
        return {"study": study.to_dict()}
    else:
        print("No such document!")
        return {"message": "No such document!", "study": {}}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       STUDY TEXT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@app.post("/update-text/")
async def update_text(text: TextUpdateRequest):
    # TODO verify user has access to study
    db = firestore.client()
    study_ref = db.collection("studies_").document(text.studyId)
    study = study_ref.get()
    if study.exists:
        study_ref.update({"text": text.text})
        return {"message": "Text updated successfully"}
    else:
        print("No such document!")
        return {"message": "No such document!"}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       UPLOAD OF RESOURCE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def delete_resource_from_storage(identifier: str):
    bucket = storage.bucket()
    blob = bucket.blob(identifier)
    blob.delete()


def upload_to_firebase_storage(file: UploadFile) -> FileUploadResponse:
    if file.filename:
        try:
            bucket = storage.bucket()
            identifier = make_file_identifier(file.filename)
            blob = bucket.blob(identifier)
            file_content = file.file.read()
            blob.upload_from_string(file_content, content_type=blob.content_type)
            url = blob.media_link
            if url:
                return FileUploadResponse(
                    file_key=identifier, file_name=file.filename, url=url
                )
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to get the file URL."
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Filename is missing.")


# TODO handle file types
# TODO verify user that user is logged in
# TODO make this into a single endpoint that takes in the file and the studyId, uploads and saves the resource to the database
# TODO REMOVE
# @app.post("/upload-resource/")
# async def upload_resource(file: UploadFile):
#     return_obj = upload_to_firebase_storage(file)
#     return return_obj


class StudyResource(BaseModel):
    studyId: str
    name: str
    url: str
    identifier: str
    category: str


class ResourceUpload(BaseModel):
    studyId: str


@app.post("/upload-resource")
async def add_resource(file: UploadFile, studyId: str):
    # TODO verfications
    # TODO return category based on file type
    upload_result = upload_to_firebase_storage(file)
    study_resource = StudyResource(
        studyId=studyId,
        identifier=upload_result.identifier,
        name=str(file.file.name),
        url=upload_result.url,
        category="pdf",  # TODO get category from file type
    )

    db = firestore.client()
    study_ref = db.collection("studies_").document(studyId)
    study = study_ref.get()
    if study.exists:
        study_ref.update({"resources": ArrayUnion([study_resource.model_dump()])})
        await prepare_resource_for_pinecone(study_resource.identifier)
        return 201
    else:
        # NOTE if the study does not exist, the resource will not be added to the database and the file should not exist in the storage
        delete_resource_from_storage(study_resource.identifier)
        return {"message": "No such document!"}, 404


# NOTE filter by catergory on the frontend
@app.post("/get-resources")
async def get_resources(studyId: str):
    # TODO verify auth
    db = firestore.client()
    doc_ref = db.collection("studies_").document(studyId)
    doc_snapshot = doc_ref.get()
    if doc_snapshot.exists:
        study_data = doc_snapshot.to_dict()
        if study_data and "resources" in study_data:
            return {"resources": study_data["resources"]}
    return {"resources": []}


####################################################
#                   CHAT                           #
####################################################
@app.post("/chat-messages")
async def chat_messages(studyId: str):
    # TODO verify user has access to study
    db = firestore.client()
    doc_ref = db.collection("studies_").document(studyId)
    doc_snapshot = doc_ref.get()
    if doc_snapshot.exists:
        study_data = doc_snapshot.to_dict()
        if study_data and "chat" in study_data:
            return {"chat_messages": study_data["chat"]}
    return {"chat_messages": []}


# TODO use streamlit to handle chat messages
@app.post("/send-chat-message")
async def send_chat_message(chat_message: AddChatMessageRequest):
    db = firestore.client()
    doc_ref = db.collection("studies_").document(chat_message.studyId)
    if not doc_ref.get().exists:
        return {"message": "No such document!"}

    if chat_message.role.lower() == "bot":
        doc_ref.update({"chat.bot": ArrayUnion([chat_message.message])})
    else:
        doc_ref.update({"chat.user": ArrayUnion([chat_message.message])})

    return {"message": "Chat message added successfully"}


@app.post("/chat")
async def chat(chat: ChatRequest, background_tasks: BackgroundTasks):
    message = chat.message
    studyId = chat.studyId
    resource_identifier = chat.resource_identifier

    background_tasks.add_task(
        save_chat_message_to_db,
        chat_message=message,
        studyId=studyId,
        role="user",
    )
    llm = OpenAI(model=LLMModel.GPT_3_5_TURBO_INSTRUCT, temperature=0, max_tokens=250)
    prompt = message

    # NOTE a bit slow
    async def generate_llm_response() -> AsyncGenerator[str, None]:
        """
        This asynchronous generator function streams the response from the language model (LLM) in chunks.
        For each chunk received from the LLM, it appends the chunk to the 'llm_response' string and yields
        the chunk to the caller. After all chunks have been received and yielded, it schedules a background task
        to save the complete response to the database as a chat message from the 'bot' role.
        """
        llm_response = ""
        async for chunk in llm.astream(prompt):
            llm_response += chunk
            yield chunk
        background_tasks.add_task(
            save_chat_message_to_db,
            chat_message=llm_response,
            studyId=studyId,
            role="bot",
        )

    return StreamingResponse(generate_llm_response())


# TODO add user verification dependency
@app.post("/get-chat-messages")
async def get_chat_messages(studyId: str):
    db = firestore.client()
    doc_ref = db.collection("studies_").document(studyId)

    doc_snapshot = doc_ref.get()
    if doc_snapshot.exists:
        study_data = doc_snapshot.to_dict()
        if study_data and "chatMessages" in study_data:
            return {"chatMessages": study_data["chatMessages"]}
    return {"chatMessages": []}


def save_chat_message_to_db(chat_message: str, studyId: str, role: str):
    db = firestore.client()
    doc_ref = db.collection("studies_").document(studyId)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="No such document!")
    new_chat_message = ChatMessage(
        role=role, content=chat_message, createdAt=datetime.now()
    ).model_dump()
    doc_ref.update({"chatMessages": ArrayUnion([new_chat_message])})


# ~~~~~~~~~~~~~~~~~~~~~~~

# TODO for the library
# @app.post("get-user-resources")
