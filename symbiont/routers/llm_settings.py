from fastapi import APIRouter, HTTPException, status, Cookie
from pydantic import BaseModel
from firebase_admin import firestore
from fastapi import Request, Response
from ..llms import get_user_llm_settings
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
    response.set_cookie(key="api_key", value=settings.api_key, samesite="None", secure=True)
    response.set_cookie(key="llm_name", value=settings.llm_name, samesite="None", secure=True)
    logger.info("LLM settings updated")

    return {"message": "LLM settings saved"}


@router.get("/get-llm-settings")
async def get_llm_settings(request: Request, api_key: Annotated[str | None, Cookie()] = None):
    user_uid = request.state.verified_user["user_id"]
    user_settings = get_user_llm_settings(user_uid)
    if user_settings is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    logger.info(f"LLM settings retrieved: {user_settings}")
    logger.info("Appending api_key to response from cookie")
    user_settings["api_key"] = api_key
    return user_settings
