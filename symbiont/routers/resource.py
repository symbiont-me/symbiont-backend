from datetime import datetime, timedelta
from io import BytesIO
from firebase_admin import storage
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, Request
from fastapi.responses import StreamingResponse

from ..models import (
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
import time
from ..mongodb import studies_collection, grid_fs, grid_fs_bucket
from ..repositories.study_resource_repo import StudyResourceRepo
import tempfile

from bson.objectid import ObjectId
from ..utils.document_loaders import load_pdf

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


# TODO handle file types


# TODO move this someplace else
async def save_summary(study_id: str, study_resource: StudyResource, content: str):
    # s = time.time()
    # summary = summarise_plain_text_resource(content)
    # logger.info("Content summarised")
    # logger.info("Now adding summary to DB")
    # if summary == "":
    #     summary = "No summary available."
    # studies_collection.update(
    #     {"_id": study_id},
    #     {"resources.identifier": study_resource.identifier},
    #     {"$set": {"resources.$.summary": summary}},
    # )
    # logger.info(f"Summary added to resource {study_resource.identifier}")
    # elapsed = time.time() - s
    # logger.info(f"Summary added in {elapsed} seconds")
    return {"message": "Summary added."}


async def upload_to_storage(file_bytes, file_identifier: str, content_type: str | None = None):
    try:
        # Create a temporary file to store the uploaded contents
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            # Write the uploaded contents to the temporary file
            temp_file.write(file_bytes)
            temp_file.flush()

            # Store the file in GridFS
            with open(temp_file.name, "rb") as f:
                file_id = grid_fs.put(f, filename=file_identifier, content_type=content_type)

        return {"file_id": str(file_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        unique_file_identifier = make_file_identifier(file.filename)

        file_bytes = await file.read()

        upload_result = await upload_to_storage(file_bytes, unique_file_identifier, file.content_type)
        # upload_result = upload_to_firebase_storage(file, user_uid)
        file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
        logger.debug(f"File uploaded to storage: {upload_result}")
        study_resource = StudyResource(
            studyId=studyId,
            identifier=unique_file_identifier,
            name=file.filename,
            url="",  # NOTE for other file types this may still be needed
            storage_ref=upload_result["file_id"],  # NOTE: this is the _id from GridFS, used for retrieval
            category=file_extension,
        )
        #
        pc_service = PineconeService(
            study_id=study_resource.studyId,
            resource_identifier=study_resource.identifier,
            user_uid=user_uid,
            user_query=None,
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(file_bytes)
            metadata = load_pdf(temp.name, unique_file_identifier)
        logger.debug(f"Pdf loader extractracted number of pages: {len(metadata)}")
        await pc_service.add_file_resource_to_pinecone(metadata)

        study_resources_repo = StudyResourceRepo(study_resource, user_id=user_uid, study_id=studyId)
        study_resources_repo.add_study_resource_to_db()

        return ResourceResponse(status_code=200, message="Resource added.", resources=[study_resource])
    except Exception as e:
        # TODO delete from storage if it fails
        logger.error(f"Error occur while adding resource: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add resource. Try Again.")


@router.get("/get-file-from-storage")
async def get_file_from_storage(storage_ref: str):
    logger.debug(f"Retrieving file from storage: {storage_ref}")
    file = grid_fs.get(ObjectId(storage_ref))
    file_content = file.read()
    # logger.debug(file_content)
    return StreamingResponse(BytesIO(file_content), media_type="application/octet-stream")
    # try:
    #     file = grid_fs.get(storage_ref)
    #     file_content = file.read()
    #     logger.debug(file_content)
    #     return file_content
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-resources")
async def get_resources(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    studies = studies_collection.find({"_id": studyId})
    if studies["userId"] != user_uid:
        raise HTTPException(
            status_code=404,
            detail="User Not Authorized to Access Study",
        )
    resources = studies["resources"]
    return ResourceResponse(
        resources=[StudyResource(**resource) for resource in resources],
        status_code=200,
        message="Resources retrieved",
    )


@router.post("/add_yt_resource")
async def add_yt_resource(
    video_resource: AddYoutubeVideoRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    if not video_resource.urls:
        raise HTTPException(status_code=400, detail="Invalid URL. Please provide a valid URL.")

    user_uid = request.state.verified_user["user_id"]
    logger.debug(f"Parsing {len(video_resource.urls)} YT Videos")
    yt_resources = []
    try:
        for url in video_resource.urls:
            loader = YoutubeLoader.from_youtube_url(
                str(url),
                add_video_info=True,
                language=["en", "id"],
                translation="en",
            )
            logger.info(f"Processing youtube video {url}")
            # @dev there should only be a single document for this
            doc = loader.load()[0]

            # @dev if the transcript is empty, the video is not processed
            # TODO if the transcript is empty, extract audio and convert to text using whisper
            if doc.page_content == "":
                raise HTTPException(
                    status_code=404,
                    detail="There is no content in the video. Please try again",
                )
            unique_file_identifier = make_file_identifier(doc.metadata["title"])

            pc_service = PineconeService(
                study_id=video_resource.studyId,
                resource_identifier=unique_file_identifier,
                user_uid=user_uid,
                user_query=None,
            )

            study_resource = StudyResource(
                studyId=video_resource.studyId,
                identifier=unique_file_identifier,
                name=doc.metadata["title"],
                url=str(url),
                category="video",
            )

            await pc_service.upload_yt_resource_to_pinecone(study_resource, doc.page_content)
            yt_resources.append(study_resource)

            # mongodb
            # NOTE should only be added to the db if the resource is successfully uploaded to Pinecone
            study_resources_repo = StudyResourceRepo(study_resource, user_id=user_uid, study_id=video_resource.studyId)
            study_resources_repo.add_study_resource_to_db()

            logger.info(f"Youtube video added to Pinecone {study_resource}")
            background_tasks.add_task(
                save_summary,
                study_resource.studyId,
                study_resource,
                doc.page_content,
            )

        return ResourceResponse(status_code=200, message="Resource added.", resources=[yt_resources])
    except Exception as e:
        logger.error(f"Error processing youtube video: {e}")
        raise HTTPException(status_code=500, detail="Error processing youtube video")


@router.post("/add-webpage-resource")
async def add_webpage_resource(
    webpage_resource: AddWebpageResourceRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid, webpage_resource.studyId)

    if not webpage_resource.urls:
        raise HTTPException(status_code=400, detail="Invalid URL. Please provide a valid URL.")

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
            docs_transformed = bs_transformer.transform_documents([doc], tags_to_extract=["p", "li", "span", "div"])
            transformed_docs_contents.append((study_resource, docs_transformed[0].page_content))
            # TODO this exception is not returning the correct status code
            if docs_transformed[0].page_content is None:
                raise HTTPException(
                    status_code=404,
                    detail="There is no content in the webpage. Please try again",
                )
            await pc_service.upload_webpage_to_pinecone(study_resource, docs_transformed[0].page_content)
            # mongodb
            study_resources_repo = StudyResourceRepo(
                study_resource, user_id=user_uid, study_id=webpage_resource.studyId
            )
            study_resources_repo.add_study_resource_to_db()

            logger.info(f"Web Resource added {study_resource}")
            background_tasks.add_task(
                save_summary,
                webpage_resource.studyId,
                study_resource,
                docs_transformed[0].page_content,
            )

        return ResourceResponse(status_code=200, message="Resource added.", resources=study_resources)
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

    # TODO rename the method as used for both plain text and webpage
    await pc_service.upload_webpage_to_pinecone(study_resource, plain_text_resource.content)

    study_resources_repo = StudyResourceRepo(study_resource, user_id=user_uid, study_id=plain_text_resource.studyId)
    study_resources_repo.add_study_resource_to_db()

    background_tasks.add_task(
        save_summary,
        plain_text_resource.studyId,
        study_resource,
        plain_text_resource.content,
    )
    return ResourceResponse(status_code=200, message="Resource added.", resources=[study_resource])


class DeleteResourceRequest(BaseModel):
    study_id: str
    identifier: str


class DeleteResourceResponse(BaseModel):
    message: str
    status_code: int
    resource: StudyResource


@router.post("/delete-resource-from-study")
async def delete_resource_from_study(delete_request: DeleteResourceRequest, request: Request):
    try:
        s = time.time()

        study_id, resource_identifier = delete_request.study_id, delete_request.identifier
        storage_ref = studies_collection.find_one(  # get the storage ref
            {"_id": study_id}, {"resources": {"$elemMatch": {"identifier": resource_identifier}}}
        )["resources"][0]["storage_ref"]

        user_uid = request.state.verified_user["user_id"]
        pc_service = PineconeService(
            study_id=str(delete_request.study_id),
            resource_identifier=resource_identifier,
            user_uid=user_uid,
        )
        # delete from pinecone
        pc_service.delete_vectors_from_pinecone(resource_identifier)

        # get the resource to delete
        resources = studies_collection.find_one(
            {"_id": delete_request.study_id}, {"resources": {"$elemMatch": {"identifier": resource_identifier}}}
        )
        resource_to_delete = resources["resources"][0]
        logger.debug(f"REsource to delete: {resource_to_delete}")
        if resource_to_delete["category"] in ["pdf", "audio", "image"]:
            # delete_resource_from_storage(user_uid, resource_identifier)
            grid_fs_bucket.delete(file_id=ObjectId(storage_ref))
            logger.info(f"Resource {resource_identifier } deleted from storage")
        # remove from db
        studies_collection.update_one({"_id": study_id}, {"$pull": {"resources": {"identifier": resource_identifier}}})
        # # remove vecotor refs from db
        studies_collection.update_one({"_id": study_id}, {"$unset": {"vectors." + resource_identifier: ""}})
        elapsed = time.time() - s
        logger.info(f"Resource deleted in {elapsed} seconds")
        #
        return DeleteResourceResponse(
            message="Resource deleted.",
            status_code=200,
            resource=resource_to_delete,
        )
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting resource")
