from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from firebase_admin import firestore, auth, credentials, storage
from fastapi import Request
from ..llms import get_user_llm_settings
from ..utils.db_utils import StudyService


router = APIRouter()


class LLMSettingsRequest(BaseModel):
    llm_name: str
    api_key: str


@router.post("/set-llm-settings")
async def set_llm_settings(settings: LLMSettingsRequest, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_service = StudyService(user_uid)
    return study_service.set_llm_settings(settings)
