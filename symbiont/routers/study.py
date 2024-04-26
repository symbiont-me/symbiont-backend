from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from ..models import CreateStudyRequest
from firebase_admin import firestore
from datetime import datetime
from ..models import Study, Chat
from .. import logger
import time
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from ..mongodb import studies_collection, users_collection
import uuid
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
    logger.info(studyId)
    s = time.time()
    logger.info("Getting current study")
    user_uid = request.state.verified_user["user_id"]
    study = studies_collection.find_one({"_id": studyId})

    if study["userId"] != user_uid:
        logger.error("User is Not authorized to access study")
        raise HTTPException(
            detail="User Not Authorized to Access Study",
            status_code=404,
        )

    elapsed = time.time() - s
    logger.info(f"Getting current study took {elapsed} seconds")
    return StudyResponse(message="", status_code=200, studies=[study])


@router.get("/get-user-studies")
async def get_user_studies(request: Request):
    user_uid = request.state.verified_user["user_id"]

    try:
        s = time.time()
        # TODO do it like this
        # 1. get study ids from the user document
        # 2. get the study data from the study collection using the ids
        # user_study_ids = users_collection.find_one({"_id": user_uid})
        # logger.info(f"User studies: {user_study_ids}")
        studies_data = list(studies_collection.find({"userId": user_uid}))
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

        result = studies_collection.insert_one({"_id": str(uuid.uuid4()), **new_study.model_dump()})
        study_data = studies_collection.find_one(result.inserted_id)

        # Add to users
        user = users_collection.find_one({"_id": user_uid})
        logger.info(f"User: {user}")
        result = users_collection.update_one({"_id": user_uid}, {"$push": {"studies": study_data["_id"]}})

        elapsed = time.time() - s
        logger.info(f"Creating study took {elapsed} seconds")
        return StudyResponse(message="Study created successfully", status_code=200, studies=[study_data])
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


@router.delete("/delete-study")
async def delete_study(studyId: str, request: Request):
    s = time.time()
    user_uid = request.state.verified_user["user_id"]
    try:
        studies = studies_collection.find_one({"_id": studyId})
        if studies["userId"] != user_uid:
            logger.error("User is Not authorized to delete study")
            raise HTTPException(
                detail="User Not Authorized to Delete Study",
                status_code=404,
            )

        studies_collection.delete_one({"_id": studyId})
        users_collection.update_one({"_id": user_uid}, {"$pull": {"studies": studyId}})

        elapsed = time.time() - s
        logger.info(f"Deleting study took {elapsed} seconds")
        return DeleteStudyResponse(message="Study deleted successfully", status_code=200, studyId=studyId)
    except Exception as e:
        logger.error(f"Error Deleting Study {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the study.",
        )
