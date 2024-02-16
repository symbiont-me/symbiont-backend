from fastapi import FastAPI
from pydantic import BaseModel
import firebase_admin
from firebase_admin import firestore, credentials
from typing import List
from pydantic.networks import HttpUrl
from google.cloud.firestore_v1 import ArrayUnion
from enum import Enum

cred = credentials.Certificate("src/serviceAccountKey.json")
firebase_admin.initialize_app(cred)


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


app = FastAPI()
# CORS policy

print(firebase_admin.get_app())


@app.get("/")
async def read_root():
    return {"Hello": "World"}


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


@app.post("/update-text/")
async def update_text(text: TextUpdateRequest):
    # TODO verify user has access to study
    db = firestore.client()
    print(text.studyId)
    study_ref = db.collection("studies_").document(text.studyId)
    study = study_ref.get()
    if study.exists:
        study_ref.update({"text": text.text})
        return {"message": "Text updated successfully"}
    else:
        print("No such document!")
        return {"message": "No such document!"}


@app.post("/add-resource/")
async def add_resource(resource: UploadResource):
    # TODO verfications
    # TODO upload to storage
    db = firestore.client()
    study_ref = db.collection("studies_").document(resource.studyId)
    study = study_ref.get()
    if study.exists:
        study_ref.update({"resources": ArrayUnion([resource.model_dump()])})
        return {"message": "Resource uploaded successfully"}
    else:
        print("No such document!")
        return {"message": "No such document!"}


@app.post("/chat-messages")
async def chat_messages(studyId: str):
    # TODO verify user has access to study
    db = firestore.client()
    doc_ref = db.collection("studies_").document(studyId)
    if not doc_ref.get().exists:
        return {"message": "No such document!"}
    # TODO fix nonscriptable error in the line below
    return {"chat_messages": doc_ref.get().to_dict()["chat"]}


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
