from fastapi import BackgroundTasks, FastAPI, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from firebase_admin import initialize_app, firestore, auth, credentials, storage
from typing import List
from pydantic.networks import HttpUrl
from enum import Enum
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import OpenAI
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


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

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
#       MODELS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LLMModel(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    GPT_4 = "gpt-4"
    LLAMA = "llama"


class FileUploadResponse(BaseModel):
    file_key: str
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


class StudyResource(BaseModel):
    studyId: str
    name: str
    url: str
    identifier: str
    category: str


class ChatRequest(BaseModel):
    message: str
    studyId: str


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
            blob.upload_from_string(
                file_content, content_type=blob.content_type)
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
@app.post("/upload-resource/")
async def upload_resource(file: UploadFile):
    return_obj = upload_to_firebase_storage(file)
    return return_obj


@app.post("/add-resource-to-db")
async def add_resource(resource: StudyResource):
    # TODO verfications
    db = firestore.client()
    study_ref = db.collection("studies_").document(resource.studyId)
    study = study_ref.get()
    if study.exists:
        study_ref.update({"resources": ArrayUnion([resource.model_dump()])})
        return 201
    else:
        # NOTE if the study does not exist, the resource will not be added to the database and the file should not exist in the storage
        delete_resource_from_storage(resource.identifier)
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

    background_tasks.add_task(
        save_chat_message_to_db,
        chat_message=message,
        studyId=studyId,
        role="user",
    )
    llm = OpenAI(model=LLMModel.GPT_3_5_TURBO_INSTRUCT,
                 temperature=0, max_tokens=250)
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
