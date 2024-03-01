from datetime import datetime, timedelta
from re import A
from google.cloud.firestore import ArrayUnion
from ..utils import make_file_identifier, verify_user_auth_token
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from firebase_admin import firestore, auth, credentials, storage
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Depends
from ..models import FileUploadResponse, GetResourcesResponse, StudyResource
from ..pinecone.pc import prepare_resource_for_pinecone
from ..utils import verify_user_auth_token

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
    print(url)
    return url


def upload_to_firebase_storage(file: UploadFile) -> FileUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")
    # TODO get user uid from token
    # TODO remove hardcoded user_uid
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
    decoded_token: dict = Depends(verify_user_auth_token),
):

    user_uid = decoded_token["uid"]
    # if user_uid is None:
    #     raise HTTPException(status_code=401, detail="Unauthorized")

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


@router.post("/get-resources")
async def get_resources(
    studyId: str, decoded_token: dict = Depends(verify_user_auth_token)
):
    user_uid = decoded_token["uid"]
    db = firestore.client()
    study_data = db.collection("studies_").where("userId", "==", user_uid).stream()

    for doc in study_data:
        if doc.id != studyId:
            continue
        study_dict = doc.to_dict()
        if study_dict is None:
            raise HTTPException(status_code=404, detail="Study not found")
        resources = study_dict.get("resources", [])
        return GetResourcesResponse(
            resources=[StudyResource(**resource) for resource in resources]
        )
    raise HTTPException(status_code=404, detail="Study not found")
