from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from ..models import CreateStudyRequest
from datetime import datetime
from ..models import Study
from .. import logger
import time
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from ..mongodb import studies_collection, users_collection
from symbiont.mongodb.utils import user_exists, check_user_authorization
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
async def get_current_study(studyId: str, request: Request) -> StudyResponse:
    try:
        s = time.time()
        logger.info("Getting current study")
        session_data = {
            "user_id": request.state.session.get_user_id(),
        }
        user_uid = session_data["user_id"]
        await user_exists(user_uid)
        check_user_authorization(studyId, user_uid, studies_collection)
        study = studies_collection.find_one({"_id": studyId})
        elapsed = time.time() - s
        logger.info(f"Getting current study took {elapsed} seconds")
        return StudyResponse(message="", status_code=200, studies=[study])

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(detail="An internal error occurred.", status_code=500)


@router.get("/get-user-studies")
async def get_user_studies(request: Request):
    try:
        s = time.time()
        session_data = {
            "user_id": request.state.session.get_user_id(),
        }
        user_uid = session_data["user_id"]
        await user_exists(user_uid)
        # We get all study ids for the user
        study_ids = users_collection.find_one({"_id": user_uid})["studies"]
        studies_data = list(studies_collection.find({"_id": {"$in": study_ids}}))

        elapsed = time.time() - s
        logger.info(f"Getting user studies took {elapsed} seconds")
        return StudyResponse(
            message="User studies retrieved successfully",
            status_code=200,
            studies=studies_data,
        )
    except Exception as e:
        logger.error(f"Error Getting User Studies {e}")
        raise HTTPException(
            detail=str(e),
            status_code=500,
        )


@router.post("/create-study")
async def create_study(study: CreateStudyRequest, request: Request):
    try:
        session_data = {
            "user_id": request.state.session.get_user_id(),
        }
        user_uid = session_data["user_id"]
        await user_exists(user_uid)
        new_study = Study(
            name=study.name,
            description=study.description,
            userId=user_uid,
            image=study.image,
            createdAt=datetime.now().isoformat(),  # Use ISO format for consistency
            resources=[],
            chat=[],
        )
        s = time.time()

        result = studies_collection.insert_one({"_id": str(uuid.uuid4()), **new_study.model_dump()})
        study_data = studies_collection.find_one(result.inserted_id)

        # Add to users
        result = users_collection.update_one({"_id": user_uid}, {"$push": {"studies": study_data["_id"]}})

        elapsed = time.time() - s
        logger.info(f"Creating study took {elapsed} seconds")
        logger.info(f"Study Details: {study_data}")
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
    try:
        session_data = {
            "user_id": request.state.session.get_user_id(),
        }

        user_uid = session_data["user_id"]
        await user_exists(user_uid)
        check_user_authorization(studyId, user_uid, studies_collection)
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
