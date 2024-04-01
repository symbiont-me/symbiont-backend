from fastapi import APIRouter, Request, HTTPException
from ..models import TextUpdateRequest
from firebase_admin import firestore
from ..utils.db_utils import StudyService

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
async def update_text(text: TextUpdateRequest, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid, text.studyId)
    study_ref = study_service.get_document_ref()
    if study_ref is None:
        raise HTTPException(status_code=404, detail="No such document!")
    study = study_ref.get()
    if study.exists:
        study_ref.update({"text": text.text})
        return {"message": "Text updated successfully"}
