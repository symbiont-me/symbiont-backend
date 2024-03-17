from datetime import datetime, timedelta
from re import A
from google.cloud.firestore import ArrayUnion, ArrayRemove
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from firebase_admin import firestore, auth, credentials, storage
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    UploadFile,
    File,
    Depends,
    Request,
)
from ..models import (
    FileUploadResponse,
    GetResourcesResponse,
    StudyResource,
    AddYoutubeVideoRequest,
    AddWebpageResourceRequest,
)
from ..pinecone.pc import PineconeService
from ..utils.db_utils import StudyService
from ..utils.helpers import make_file_identifier
from langchain_community.document_loaders import YoutubeLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from ..utils.llm_utils import summarise_plain_text_resource
from pydantic import BaseModel
from ..pinecone.pc import PineconeService


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#      RESOURCE UPLOAD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

router = APIRouter()


def delete_resource_from_storage(identifier: str):
    bucket = storage.bucket()
    blob = bucket.blob(identifier)
    blob.delete()


# Generates a signed URL is a URL that includes a signature, allowing access to a resource for a limited time period.
# This is useful for providing short-term access to a resource that would otherwise require authentication.
def generate_signed_url(identifier: str) -> str:
    blob = storage.bucket().blob(identifier)
    expiration_time = datetime.now() + timedelta(hours=1)
    url = blob.generate_signed_url(expiration=expiration_time, method="GET")
    return url


def upload_to_firebase_storage(file: UploadFile, user_id: str) -> FileUploadResponse:
    """
    Uploads a file to Firebase Storage.

    Args:
        file (UploadFile): The file to be uploaded.
        user_id (str): The ID of the user.

    Returns:
        FileUploadResponse: The response containing the uploaded file details.

    Raises:
        HTTPException: If the filename is missing or if there is an error during the upload process.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")
    try:
        bucket = storage.bucket()
        file_name = make_file_identifier(file.filename)
        identifier = f"userFiles/{user_id}/{file_name}"

        blob = bucket.blob(identifier)

        file_content = file.file.read()
        # TODO handle content types properly
        # NOTE this prevents the file from being downloaded in the browser if the content type is not set properly
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


# TODO handle file types


# NOTE I don't like this
# TODO move this someplace else
async def get_and_save_summary_to_db(
    study_resource: StudyResource, content: str, studyId: str, user_uid: str
):
    # TODO Fix summariser
    # TODO add the resource to db straight away
    study_service = StudyService(user_uid, studyId)
    study_service.add_resource_to_db(study_resource)
    # summary = summarise_plain_text_resource(content)
    # study_service.update_resource_summary(study_resource.identifier, summary)


@router.post("/upload-resource")
async def add_resource(
    file: UploadFile,
    studyId: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided!")
    pc_service = PineconeService(request.state.verified_user["user_id"])
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid, studyId)
    upload_result = upload_to_firebase_storage(file, user_uid)
    file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
    study_resource = StudyResource(
        studyId=studyId,
        identifier=upload_result.identifier,
        name=upload_result.file_name,
        url=upload_result.url,
        category=file_extension,
    )
    study_ref = study_service.get_document_ref()
    if study_ref is None:
        # NOTE if the study does not exist, the resource will not be added to the database and the file should not exist in the storage
        delete_resource_from_storage(study_resource.identifier)
        raise HTTPException(status_code=404, detail="No such document!")
    study_ref.update({"resources": ArrayUnion([study_resource.model_dump()])})
    print("Adding to Pinecone")
    await pc_service.prepare_resource_for_pinecone(
        upload_result.identifier, upload_result.download_url
    )
    # Add background task to generate summary
    background_tasks.add_task(
        get_and_save_summary_to_db,
        study_resource,
        upload_result.download_url,
        studyId,
        user_uid,
    )
    return {"resource": study_resource.model_dump(), "status_code": 200}


@router.post("/get-resources")
async def get_resources(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid, studyId)
    study_dict = study_service.get_document_dict()
    if study_dict is None:
        raise HTTPException(status_code=404, detail="Study not found")
    resources = study_dict.get("resources", [])
    return GetResourcesResponse(
        resources=[StudyResource(**resource) for resource in resources]
    )


@router.post("/process-youtube-video")
async def process_youtube_video(
    video_resource: AddYoutubeVideoRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    pc_service = PineconeService(request.state.verified_user["user_id"])
    # TODO allow multiple urls
    # TODO auth verification if necessary
    loader = YoutubeLoader.from_youtube_url(
        str(video_resource.url),
        add_video_info=True,
        language=["en", "id"],
        translation="en",
    )

    print("PARSING YT VIDEO")

    # @dev there should only be a single document for this
    doc = loader.load()[0]
    study_resource = StudyResource(
        studyId=video_resource.studyId,
        identifier=make_file_identifier(doc.metadata["title"]),
        name=doc.metadata["title"],
        url=str(video_resource.url),
        category="video",
    )
    background_tasks.add_task(
        get_and_save_summary_to_db,
        study_resource,
        doc.page_content,
        video_resource.studyId,
        request.state.verified_user["user_id"],
    )

    await pc_service.upload_yt_resource_to_pinecone(study_resource, doc.page_content)
    print("YT VIDEO ADDED TO PINECONE")
    return {"status_code": 200, "message": "Resource added."}


@router.post("/add-webpage-resource")
async def add_webpage_resource(
    webpage_resource: AddWebpageResourceRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):

    user_uid = request.state.verified_user["user_id"]
    pc_service = PineconeService(user_uid)
    study_service = StudyService(user_uid, webpage_resource.studyId)
    # user_uid = "U38yTj1YayfqZgUNlnNcKZKNCVv2"

    loader = AsyncHtmlLoader([str(url) for url in webpage_resource.urls])
    html_docs = loader.load()
    studies = []
    transformed_docs_contents = []  # Collect transformed docs content here
    print("PARSING WEBPAGE")
    print(html_docs)
    for index, doc in enumerate(html_docs):
        identifier = make_file_identifier(doc.metadata["title"])
        print(doc)
        study_resource = StudyResource(
            studyId=webpage_resource.studyId,
            identifier=identifier,
            name=doc.metadata["title"],
            url=str(webpage_resource.urls[index]),  # Assign URL based on index
            category="webpage",
            summary="",
        )
        studies.append(study_resource)
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            [doc], tags_to_extract=["p", "li", "span", "div"]
        )
        transformed_docs_contents.append(
            (study_resource, docs_transformed[0].page_content)
        )

        background_tasks.add_task(
            pc_service.upload_webpage_to_pinecone,
            study_resource,
            docs_transformed[0].page_content,
        )

    # Process summaries and db update as a background task
    for study_resource, content in transformed_docs_contents:
        background_tasks.add_task(
            get_and_save_summary_to_db,
            study_resource,
            content,
            webpage_resource.studyId,
            user_uid,
        )


class AddPlainTextResourceRequest(BaseModel):
    studyId: str
    name: str
    content: str


@router.post("/add-plain-text-resource")
async def add_plain_text_resource(
    plain_text_resource: AddPlainTextResourceRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    user_uid = request.state.verified_user["user_id"]

    # user_uid = "U38yTj1YayfqZgUNlnNcKZKNCVv2"
    study_resource = StudyResource(
        studyId=plain_text_resource.studyId,
        identifier=make_file_identifier(plain_text_resource.name),
        name=plain_text_resource.name,
        url="",
        category="text",
        summary="",
    )
    pc_service = PineconeService(
        user_uid,
        plain_text_resource.content,
        study_resource.identifier,
        plain_text_resource.studyId,
    )
    background_tasks.add_task(
        get_and_save_summary_to_db,
        study_resource,
        plain_text_resource.content,
        plain_text_resource.studyId,
        user_uid,
    )
    await pc_service.upload_webpage_to_pinecone(
        study_resource, plain_text_resource.content
    )
    return {"status_code": 200, "message": "Resource added."}


# TODO FIX !TRASH CODE
@router.post("/delete-resource")
async def delete_resource(identifier: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    pc_service = PineconeService(user_uid)
    # TODO fix this
    study_docs = (
        firestore.client()
        .collection("studies")
        .where("userId", "==", user_uid)
        .stream()
    )
    resource = None
    for doc in study_docs:
        doc_dict = doc.to_dict()

        resources = doc_dict.get("resources", [])
        for res in resources:
            if res.get("identifier") == identifier:
                resources.remove(res)
                doc.reference.update({"resources": resources})
                if res.get("category") == "pdf":
                    delete_resource_from_storage(identifier)
                    pc_service.delete_vectors_from_pinecone(identifier)
                return {"message": "Resource deleted."}
    raise HTTPException(status_code=404, detail="Resource not found")
    return {"message": "Resource deleted."}
