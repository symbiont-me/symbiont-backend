from datetime import datetime, timedelta
from google.cloud.firestore import ArrayUnion
from firebase_admin import firestore, storage
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    UploadFile,
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
from langchain_community.document_transformers import BeautifulSoupTransformer
from pydantic import BaseModel
from .. import logger
from ..fb.storage import delete_local_file
from ..utils.llm_utils import summarise_plain_text_resource
import time
from pydantic import BaseModel

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#      RESOURCE UPLOAD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

router = APIRouter()


class ResourceResponse(BaseModel):
    status_code: int
    message: str
    resources: list


def delete_resource_from_storage(user_uid: str, identifier: str):
    bucket = storage.bucket()
    storage_url = f"userFiles/{user_uid}/{identifier}"
    blob = bucket.blob(storage_url)
    blob.delete()
    logger.info(f"Resource deleted from storage: userFiles/userId/{identifier}")


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
        unique_file_identifier = make_file_identifier(file.filename)
        storage_path = f"userFiles/{user_id}/{unique_file_identifier}"

        blob = bucket.blob(storage_path)

        file_content = file.file.read()
        # TODO handle content types properly
        # NOTE this prevents the file from being downloaded in the browser if the content type is not set properly
        content_type = ""
        if file.filename.endswith(".pdf"):
            content_type = "application/pdf"
        blob.upload_from_string(file_content, content_type=content_type)
        url = blob.media_link
        download_url = generate_signed_url(storage_path)
        if url:
            return FileUploadResponse(
                identifier=unique_file_identifier,
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
async def save_summary(study_id: str, study_resource: StudyResource, content: str):
    s = time.time()
    summary = summarise_plain_text_resource(content)
    logger.info("Content summarised")
    logger.info("Now adding summary to DB")
    if summary == "":
        summary = "No summary available."
    db = firestore.client()
    study_ref = db.collection("studies").document(study_id)
    study_dict = study_ref.get().to_dict()
    if study_dict is None:
        raise HTTPException(status_code=404, detail="Study not found")
    resources = study_dict.get("resources", [])
    for resource in resources:
        if resource.get("identifier") == study_resource.identifier:
            resource["summary"] = summary
            study_ref.update({"resources": resources})
            logger.info(f"Summary added to resource {study_resource.identifier}")
            elapsed = time.time() - s
            logger.info(f"Summary added in {elapsed} seconds")
            return {"message": "Summary added."}


@router.post("/upload-resource")
async def add_resource(
    file: UploadFile,
    studyId: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided!")
    user_uid = request.state.verified_user["user_id"]
    try:
        upload_result = upload_to_firebase_storage(file, user_uid)
        file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
        logger.info(f"File resource uploaded to Firebase")

        study_resource = StudyResource(
            studyId=studyId,
            identifier=upload_result.identifier,
            name=upload_result.file_name,
            url=upload_result.url,  # TODO should be view_url for clarity here and in the frontend
            category=file_extension,
        )

        study_service = StudyService(user_uid, studyId)
        pc_service = PineconeService(
            study_id=study_resource.studyId,
            resource_identifier=study_resource.identifier,
            user_uid=user_uid,
            user_query=None,
            resource_download_url=upload_result.download_url,
        )
        await pc_service.add_file_resource_to_pinecone()
        study_service.add_resource_to_db(study_resource)
        return ResourceResponse(
            status_code=200, message="Resource added.", resources=[study_resource]
        )
    except Exception as e:
        # TODO delete from storage if it fails
        logger.error(f"Error occur while adding resource: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to add resource. Try Again."
        )


@router.post("/get-resources")
async def get_resources(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid, studyId)
    study_dict = study_service.get_document_dict()
    if study_dict is None:
        raise HTTPException(status_code=404, detail="Study not found")
    resources = study_dict.get("resources", [])
    logger.debug(f"Resources: {resources}")
    return ResourceResponse(
        resources=[StudyResource(**resource) for resource in resources],
        status_code=200,
        message="Resources retrieved",
    )


@router.post("/process-youtube-video")
async def process_youtube_video(
    video_resource: AddYoutubeVideoRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    try:
        user_uid = request.state.verified_user["user_id"]

        # TODO allow multiple urls
        # TODO auth verification if necessary
        loader = YoutubeLoader.from_youtube_url(
            str(video_resource.url),
            add_video_info=True,
            language=["en", "id"],
            translation="en",
        )
        logger.info(f"Processing youtube video {video_resource.url}")
        # @dev there should only be a single document for this
        doc = loader.load()[0]
        # @dev if the transcript is empty, the video is not processed
        # TODO if the transcript is empty, extract audio and convert to text using whisper
        if doc.page_content == "":
            raise HTTPException(
                status_code=404,
                detail="There is no content in the video. Please try again",
            )
        study_resource = StudyResource(
            studyId=video_resource.studyId,
            identifier=make_file_identifier(doc.metadata["title"]),
            name=doc.metadata["title"],
            url=str(video_resource.url),
            category="video",
        )
        pc_service = PineconeService(
            study_id=video_resource.studyId,
            resource_identifier=study_resource.identifier,
            user_uid=user_uid,
            user_query=None,
        )

        study_service = StudyService(user_uid, video_resource.studyId)

        await pc_service.upload_yt_resource_to_pinecone(
            study_resource, doc.page_content
        )
        # NOTE should only be added to the db if the resource is successfully uploaded to Pinecone
        study_service.add_resource_to_db(study_resource)

        logger.info(f"Youtube video added to Pinecone {study_resource}")
        background_tasks.add_task(
            save_summary,
            study_resource.studyId,
            study_resource,
            doc.page_content,
        )

        return ResourceResponse(
            status_code=200, message="Resource added.", resources=[study_resource]
        )
    except Exception as e:
        logger.error(f"Error processing youtube video: {e}")
        # TODO delete from db if it fails
        raise HTTPException(status_code=500, detail="Error processing youtube video")


@router.post("/add-webpage-resource")
async def add_webpage_resource(
    webpage_resource: AddWebpageResourceRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid, webpage_resource.studyId)
    # user_uid = "U38yTj1YayfqZgUNlnNcKZKNCVv2"
    try:
        loader = AsyncHtmlLoader([str(url) for url in webpage_resource.urls])
        html_docs = loader.load()
        study_resources = []
        transformed_docs_contents = []  # Collect transformed docs content here
        logger.info(f"Processing webpage {webpage_resource.urls}")
        logger.info(f"Parsing {len(html_docs)} documents")
        for index, doc in enumerate(html_docs):
            identifier = make_file_identifier(doc.metadata["title"])
            study_resource = StudyResource(
                studyId=webpage_resource.studyId,
                identifier=identifier,
                name=doc.metadata["title"],
                url=str(webpage_resource.urls[index]),  # Assign URL based on index
                category="webpage",
                summary="",
            )
            pc_service = PineconeService(
                study_id=webpage_resource.studyId,
                resource_identifier=identifier,
                user_uid=user_uid,
            )
            study_resources.append(study_resource)
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(
                [doc], tags_to_extract=["p", "li", "span", "div"]
            )
            transformed_docs_contents.append(
                (study_resource, docs_transformed[0].page_content)
            )
            await pc_service.upload_webpage_to_pinecone(
                study_resource, docs_transformed[0].page_content
            )
            study_service.add_resource_to_db(study_resource)
            logger.info(f"Web Resource added {study_resource}")
            background_tasks.add_task(
                save_summary,
                webpage_resource.studyId,
                study_resource,
                docs_transformed[0].page_content,
            )

        return ResourceResponse(
            status_code=200, message="Resource added.", resources=study_resources
        )
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing webpage")


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
        study_id=plain_text_resource.studyId,
        user_uid=user_uid,
        resource_identifier=study_resource.identifier,
    )
    study_service = StudyService(user_uid, plain_text_resource.studyId)

    # TODO rename the method as used for both plain text and webpage
    await pc_service.upload_webpage_to_pinecone(
        study_resource, plain_text_resource.content
    )

    study_service.add_resource_to_db(study_resource)
    background_tasks.add_task(
        save_summary,
        plain_text_resource.studyId,
        study_resource,
        plain_text_resource.content,
    )
    return ResourceResponse(
        status_code=200, message="Resource added.", resources=[study_resource]
    )


class DeleteResourceRequest(BaseModel):
    identifier: str


class DeleteResourceResponse(BaseModel):
    message: str
    status_code: int


# TODO refactor
def remove_resource_and_vector(user_uid: str, study_id: str, resource_identifier: str):
    """
    Removes a resource and its associated vector from the database.

    Args:
        study_id (str): The ID of the study.
        resource_identifier (str): The identifier of the resource to be removed.
    """
    db = firestore.client()
    study_ref = db.collection("studies").document(study_id)
    study_snapshot = study_ref.get()

    if not study_snapshot.exists:
        logger.error("The specified study does not exist.")
        raise HTTPException(status_code=404, detail="Study not found")

    study_data = study_snapshot.to_dict()
    if study_data is None:
        raise HTTPException(status_code=404, detail="Study not found")

    study_resources = study_data.get("resources", [])
    resource_vectors = study_data.get("vectors", {})

    # Remove resource from the study's resources list
    updated_study_resources = []
    for resource in study_resources:
        if resource.get("identifier") != resource_identifier:
            updated_study_resources.append(resource)
        if resource.get("category") in ["pdf", "audio", "jpg", "png"]:
            delete_resource_from_storage(user_uid, resource_identifier)
    study_resources = updated_study_resources
    study_ref.update({"resources": study_resources})
    logger.info(f"Resource {resource_identifier} deleted from study {study_id}")

    # Remove resource vector from the study's vectors
    if resource_vectors.get(resource_identifier):
        del resource_vectors[resource_identifier]
        study_ref.update({"vectors": resource_vectors})
        logger.info(
            f"Resource vector {resource_identifier} deleted from study {study_id}"
        )


@router.post("/delete-resource")
async def delete_resource(delete_request: DeleteResourceRequest, request: Request):
    s = time.time()
    logger.debug(f"Deleting resource {delete_request.identifier}")
    identifier = delete_request.identifier
    user_uid = request.state.verified_user["user_id"]
    pc_service = PineconeService(
        study_id="",
        resource_identifier=identifier,
        user_uid=user_uid,
    )
    db = firestore.client()
    user_studies_list = (
        db.collection("users").document(user_uid).get().to_dict().get("studies", [])
    )

    logger.debug(f"Deleting resource {identifier} from studies {user_studies_list}")
    for study_id in user_studies_list:
        remove_resource_and_vector(user_uid, study_id, identifier)
        pc_service.delete_vectors_from_pinecone(identifier)
    elapsed = time.time() - s
    logger.info(f"Resource deleted in {elapsed} seconds")
    return {"message": "Resource deleted."}
