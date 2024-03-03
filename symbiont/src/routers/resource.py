from datetime import datetime, timedelta
from google.cloud.firestore import ArrayUnion
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
    ProcessYoutubeVideoRequest,
    ProcessWebpageResourceRequest,
)
from ..pinecone.pc import (
    prepare_resource_for_pinecone,
    upload_yt_resource_to_pinecone,
    upload_webpage_to_pinecone,
)
from typing import Optional
from ..utils.db_utils import get_document_dict, get_document_ref
from ..utils.helpers import make_file_identifier
from langchain_community.document_loaders import YoutubeLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from ..utils.llm_utils import summarise_webpage_resource

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#      RESOURCE UPLOAD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

router = APIRouter()


def delete_resource_from_storage(identifier: str):
    bucket = storage.bucket()
    blob = bucket.blob(identifier)
    blob.delete()


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
# TODO verify user that user is logged in
# TODO make this into a single endpoint that takes in the file and the studyId, uploads and saves the resource to the database
# TODO REMOVE
# @app.post("/upload-resource/")
# async def upload_resource(file: UploadFile):
#     return_obj = upload_to_firebase_storage(file)
#     return return_obj


@router.post("/upload-resource")
async def add_resource(
    file: UploadFile,
    studyId: str,
    request: Request,
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided!")
    user_uid = request.state.verified_user["user_id"]
    upload_result = upload_to_firebase_storage(file, user_uid)
    file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
    study_resource = StudyResource(
        studyId=studyId,
        identifier=upload_result.identifier,
        name=upload_result.file_name,
        url=upload_result.url,
        category=file_extension,
    )
    study_ref = get_document_ref("studies_", "userId", user_uid, studyId)
    if study_ref is None:
        # NOTE if the study does not exist, the resource will not be added to the database and the file should not exist in the storage
        delete_resource_from_storage(study_resource.identifier)
        raise HTTPException(status_code=404, detail="No such document!")
    study_ref.update({"resources": ArrayUnion([study_resource.model_dump()])})
    print("Adding to Pinecone")
    await prepare_resource_for_pinecone(
        upload_result.identifier, upload_result.download_url
    )
    return {"resource": study_resource.model_dump()}


@router.post("/get-resources")
async def get_resources(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_dict = get_document_dict("studies_", "userId", user_uid, studyId)
    if study_dict is None:
        raise HTTPException(status_code=404, detail="Study not found")
    resources = study_dict.get("resources", [])
    return GetResourcesResponse(
        resources=[StudyResource(**resource) for resource in resources]
    )
