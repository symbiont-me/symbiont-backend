from fastapi import BackgroundTasks, FastAPI, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from firebase_admin import initialize_app, firestore, auth, credentials, storage
from typing import List, Literal
from pydantic.networks import HttpUrl
from enum import Enum
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import OpenAI, OpenAIEmbeddings
import os
from langchain.chains import LLMChain
from dotenv import load_dotenv
from datetime import datetime, timedelta
from google.cloud.firestore import ArrayUnion
from .utils import (
    make_file_identifier,
    verify_token,
)
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from hashlib import md5
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from typing import AsyncGenerator, Optional

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


async def download_from_firebase_storage(
    file_key: str, download_url: str
) -> str | None:
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        save_path = f"temp/{file_key}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully and saved to {save_path}")
        return save_path
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


# TODO maybe delete the entire directory because we do not want to keep the user id in the file path
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


async def handle_pdf_resource(file_path: str):
    if file_path is not None and file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = [PdfPage(**page.dict()) for page in loader.load_and_split()]

        docs = []
        for page in pages:
            prepared_pages = await prepare_pdf_for_pinecone(page)
            docs.extend(prepared_pages)
        vecs = [await embed_document(doc) for doc in docs]
        return vecs
    return []


async def prepare_resource_for_pinecone(file_identifier: str, download_url: str):
    file_path = await download_from_firebase_storage(file_identifier, download_url)
    # handle pdf only for now
    if file_path is not None and file_path.endswith(".pdf"):
        vecs = await handle_pdf_resource(file_path)
        await upload_vecs_to_pinecone(vecs, file_identifier)
        await delete_local_file(file_path)


async def upload_vecs_to_pinecone(vecs: List[PineconeRecord], file_identifier: str):
    # TODO don't initialize Pinecone client here
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


@app.post("/update-text")
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


def generate_signed_url(identifier: str) -> str:
    blob = storage.bucket().blob(identifier)
    expiration_time = datetime.now() + timedelta(hours=1)
    url = blob.generate_signed_url(expiration=expiration_time, method="GET")
    print(url)
    return url


def upload_to_firebase_storage(file: UploadFile) -> FileUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")
    user_uid = "U38yTj1YayfqZgUNlnNcKZKNCVv2"
    try:
        bucket = storage.bucket()
        file_name = make_file_identifier(file.filename)
        identifier = f"userFiles/{user_uid}/{file_name}"

        blob = bucket.blob(identifier)

        file_content = file.file.read()
        # TODO handle content types properly
        content_type = ""
        if file.filename.endswith(".pdf"):
            content_type = "application/pdf"
        blob.upload_from_string(file_content, content_type=content_type)
        url = blob.media_link
        download_url = generate_signed_url(identifier)
        if url:
            return FileUploadResponse(
                identifier=identifier,
                file_name=file.filename,
                url=url,
                download_url=download_url,
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to get the file URL.")
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
        name=upload_result.file_name,
        url=upload_result.url,
        category="pdf",  # TODO get category from file type
    )

    db = firestore.client()
    study_ref = db.collection("studies_").document(studyId)
    study = study_ref.get()
    if study.exists:
        study_ref.update({"resources": ArrayUnion([study_resource.model_dump()])})
        print("Adding to Pinecone")
        await prepare_resource_for_pinecone(
            upload_result.identifier, upload_result.download_url
        )
        return {"resource": study_resource.model_dump()}

    else:
        # NOTE if the study does not exist, the resource will not be added to the database and the file should not exist in the storage
        delete_resource_from_storage(study_resource.identifier)
        return {"message": "No such document!"}, 404


class GetResourcesResponse(BaseModel):
    resources: List[StudyResource]


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
            resources = [
                StudyResource(**resource) for resource in study_data["resources"]
            ]
            # returns the StudyResource objects
            return GetResourcesResponse(resources=resources)
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


class Message(BaseModel):
    content: str
    createdAt: datetime
    role: Literal["user", "bot"]


class ChatRequest(BaseModel):
    user_query: str
    previous_message: str
    study_id: str
    resource_identifier: Optional[str] = None


class UserQuery(BaseModel):
    query: str
    previous_message: str | None
    study_id: str


@app.post("/mock-chat")
async def mock_chat(chat: ChatRequest):
    # Example logic to process the chat request
    # This is where you'd integrate your chat logic, AI model, etc.
    # For demonstration, let's just echo back the user_query with some additional text
    response_text = f"Received your query"
    print("User asked:", chat)
    # Construct and return the response
    # You might want to return more structured data depending on your frontend needs
    return {"response": response_text}


@app.post("/chat")
async def chat(chat: ChatRequest, background_tasks: BackgroundTasks):
    print(chat.resource_identifier)
    # if chat.resource_identifier:
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Resource identifier is not supported in this version of the chat endpoint",
    #     )
    user_query = chat.user_query
    previous_message = chat.previous_message
    study_id = chat.study_id
    resource_identifier = chat.resource_identifier
    background_tasks.add_task(
        save_chat_message_to_db,
        chat_message=user_query,
        studyId=study_id,
        role="user",
    )
    llm = OpenAI(
        model=LLMModel.GPT_3_5_TURBO_INSTRUCT, temperature=0.75, max_tokens=1500
    )
    context = ""

    if resource_identifier:
        context = get_chat_context(user_query, resource_identifier)
    print("Context:", context)
    complete_context = context + " " + previous_message

    base_prompt = (
        "The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness. "
        "AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation. "
        "AI assistant will take into account any DOCUMENT BLOCK that is provided in a conversation.\n\n"
        "START DOCUMENT BLOCK\n\n" + complete_context + "\n\nEND OF DOCUMENT BLOCK\n\n"
        "If the context does not provide the answer to the question or the context is empty, the AI assistant will say, "
        "\"I'm sorry, but I don't know the answer to that question\". "
        "AI assistant will not invent anything that is not drawn directly from the context. "
        "AI will keep answers short and to the point. "
        "AI will return the response in valid Markdown format."
    )
    prompt = f"${base_prompt} ${previous_message} ${user_query}"

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
            studyId=study_id,
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
