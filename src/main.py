from fastapi import FastAPI, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from firebase_admin import initialize_app, firestore, auth, credentials, storage
from typing import List
from pydantic.networks import HttpUrl
from google.cloud.firestore_v1 import ArrayUnion
from enum import Enum
from typing import Optional
from fastapi.responses import JSONResponse

# from .fb.storage import save_file


app = FastAPI()
# CORS policy

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
cred = credentials.Certificate("src/serviceAccountKey.json")
initialize_app(cred, {"storageBucket": "symbiont-e7f06.appspot.com"})


def remove_non_ascii(text):
    return "".join(i for i in text if ord(i) < 128)


def replace_space_with_underscore(text):
    return text.replace(" ", "_")


def clean_filename(text):
    return remove_non_ascii(replace_space_with_underscore(text))


def save_file(file: UploadFile):
    bucket = storage.bucket()
    file.filename = clean_filename(file.filename)
    blob = bucket.blob(file.filename)
    file_content = file.file.read()
    blob.upload_from_string(file_content, content_type=blob.content_type)
    url = blob.media_link
    # TODO return identifier
    return url


async def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    try:
        id_token = authorization.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


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


@app.post("/upload-resource/")
async def upload_resource(file: UploadFile):

    save_file(file)
    return {"message": "File uploaded successfully"}


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


# TODO for the library


# @app.post("get-user-resources")
