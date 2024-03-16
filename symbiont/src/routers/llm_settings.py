from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from firebase_admin import firestore, auth, credentials, storage
from fastapi import Request
from ..llms import get_user_llm_settings


router = APIRouter()


class LLMSettingsRequest(BaseModel):
    llm_name: str
    api_key: str


@router.post("/set-llm-settings")
async def set_llm_settings(settings: LLMSettingsRequest, request: Request):
    user_uid = request.state.verified_user["user_id"]
    db = firestore.client()
    doc_ref = db.collection("users").document(user_uid)
    if not doc_ref:
        return {"error": "User not found", "status_code": 404}
    doc_ref.set(
        {"settings": {"llm_name": settings.llm_name, "api_key": settings.api_key}}
    )

    return {"message": "LLM settings updated"}
