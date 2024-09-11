from fastapi import APIRouter, Request, HTTPException
from ..models import TextUpdateRequest
from ..mongodb import studies_collection
from symbiont.mongodb.utils import user_exists, check_user_authorization

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       STUDY TEXT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

router = APIRouter()


@router.get("/get-text")
async def get_text(study_id: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid, study_id)
    study_dict = study_service.get_document_dict()
    if study_dict is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return {"text": study_dict.get("text", "")}


@router.post("/update-text")
async def update_text(text_request: TextUpdateRequest, request: Request):
    session_data = {
        "user_id": request.state.session.get_user_id(),
    }

    user_uid = session_data["user_id"]
    await user_exists(user_uid)
    check_user_authorization(text_request.studyId, user_uid, studies_collection)
    studies_collection.update_one(
        {"_id": text_request.studyId}, {"$set": {"text": text_request.text}}
    )
    return {"message": "Text updated successfully"}
