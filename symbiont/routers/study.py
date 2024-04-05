from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from ..models import CreateStudyRequest
from firebase_admin import firestore
from ..utils.db_utils import StudyService
from datetime import datetime
from ..models import Study, Chat
from ..utils.db_utils import UserService
from .. import logger
import time
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       USER STUDIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


router = APIRouter()


class StudyResponse(BaseModel):
    message: str
    status_code: int
    studies: Optional[List[Dict[str, Any]]]


@router.get("/get-current-study")
async def get_current_study(studyId: str, request: Request):
    s = time.time()
    logger.info("Getting current study")
    user_uid = request.state.verified_user["user_id"]
    db = firestore.client()
    study_ref = db.collection("studies").document(studyId)
    study = study_ref.get().to_dict()
    if study is None:
        logger.error("Study does not exist.")
        raise HTTPException(status_code=404, detail="Study does not exist.")

    user_doc = db.collection("users").document(user_uid).get().to_dict()
    if user_doc is None:
        logger.error("User does not exist.")
        raise HTTPException(status_code=404, detail="User does not exist.")
    user_studies = user_doc.get("studies", [])
    if studyId not in user_studies:
        logger.error("User does not have access to this study.")
        raise HTTPException(
            status_code=403, detail="User does not have access to this study."
        )
    elapsed = time.time() - s
    logger.info(f"Getting current study took {elapsed} seconds")
    return StudyResponse(message="", status_code=200, studies=[study])


@router.get("/get-user-studies")
async def get_user_studies(request: Request):
    user_uid = request.state.verified_user["user_id"]

    try:
        s = time.time()
        db = firestore.client()
        studies_ref = db.collection("studies")
        query = studies_ref.where("userId", "==", user_uid)
        studies = query.stream()
        # Create a list of dictionaries, each containing the studyId and the study's data
        studies_data = [
            {"id": study.id, **(study.to_dict() or {})} for study in studies
        ]
        elapsed = time.time() - s
        logger.info(f"Getting user studies took {elapsed} seconds")
        return StudyResponse(
            message="User studies retrieved successfully",
            status_code=200,
            studies=studies_data,
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=500,
        )


@router.post("/create-study")
async def create_study(study: CreateStudyRequest, request: Request):
    user_uid = request.state.verified_user["user_id"]

    new_study = Study(
        name=study.name,
        description=study.description,
        userId=user_uid,
        image=study.image,
        createdAt=datetime.now().isoformat(),  # Use ISO format for consistency
        resources=[],
        chat=Chat(),
    )
    try:
        s = time.time()
        db = firestore.client()
        batch = db.batch()

        # Create a new document for the study
        study_doc_ref = db.collection("studies").document()
        batch.set(study_doc_ref, new_study.model_dump())

        # Get the user document and update the studies array
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            user_studies = user_doc.to_dict().get("studies", [])
            user_studies.append(study_doc_ref.id)
            batch.update(user_doc_ref, {"studies": user_studies})
        else:
            # If the user does not exist, create a new document with the study
            batch.set(user_doc_ref, {"studies": [study_doc_ref.id]})

        batch.commit()
        elapsed = time.time() - s
        logger.info(f"Creating study took {elapsed} seconds")
        return StudyResponse(
            message="Study created successfully",
            status_code=200,
            studies=[{"id": study_doc_ref.id, **new_study.model_dump()}],
        )
    except Exception as e:
        logger.error(f"Error Creating New Study {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred while creating the study.",
                "details": str(e),
            },
        )


class DeleteStudyResponse(BaseModel):
    message: str
    status_code: int
    studyId: str


# NOTE a bit slow
@router.delete("/delete-study")
async def delete_study(studyId: str, request: Request):
    s = time.time()
    user_uid = request.state.verified_user["user_id"]
    db = firestore.client()
    # Get references to the study and user documents
    study_doc_ref = db.collection("studies").document(studyId)
    user_doc_ref = db.collection("users").document(user_uid)

    # Check if the study exists
    study_doc = study_doc_ref.get()
    if not study_doc.exists:
        raise HTTPException(status_code=404, detail="Study does not exist.")

    # Check if the user exists and has the study listed
    user_doc = user_doc_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User does not exist.")
    user_studies = user_doc.to_dict().get("studies", [])
    if studyId not in user_studies:
        raise HTTPException(
            status_code=403, detail="Study not found in user's studies."
        )

    # Remove the study from the user's list of studies
    user_studies.remove(studyId)
    user_doc_ref.update({"studies": user_studies})
    # Delete the study document as well
    study_doc_ref.delete()

    elapsed = time.time() - s
    logger.info(f"Deleting study took {elapsed} seconds")
    return DeleteStudyResponse(
        message="Study deleted successfully", status_code=200, studyId=studyId
    )
