from fastapi import APIRouter, Depends, HTTPException, status, Cookie
from pydantic import BaseModel
from firebase_admin import firestore, auth, credentials, storage
from fastapi import Request, Response
from ..llms import get_user_llm_settings
from ..utils.db_utils import StudyService
from typing import Annotated
from .. import logger

router = APIRouter()


class LLMSettingsRequest(BaseModel):
    llm_name: str
    api_key: str


@router.post("/set-llm-settings")
async def set_llm_settings(
    settings: LLMSettingsRequest,
    request: Request,
    response: Response,
):
    db = firestore.client()
    user_uid = request.state.verified_user["user_id"]
    user_doc = db.collection("users").document(user_uid)
    user_data = user_doc.get().to_dict()
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    user_settings = user_data.get("settings")
    if user_settings is None:
        user_settings = {}

    # @note don't save the api_key in the database
    user_settings["llm_name"] = settings.llm_name
    user_doc.update({"settings": user_settings})
    response.set_cookie(
        key="api_key", value=settings.api_key, samesite="None", secure=True
    )
    response.set_cookie(
        key="llm_name", value=settings.llm_name, samesite="None", secure=True
    )
    return {"message": "LLM settings saved"}
