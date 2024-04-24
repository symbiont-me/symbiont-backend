from fastapi import APIRouter, HTTPException, status, Cookie
from pydantic import BaseModel
from fastapi import Request, Response
from typing import Annotated
from .. import logger
from ..mongodb import user_collection


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
    user_uid = request.state.verified_user["user_id"]
    # we attach the api_key to the response as we are not storing it in the database
    response.set_cookie(key="api_key", value=settings.api_key, samesite="None", secure=True)
    # delete the api_key from the settings object for security
    del settings.api_key
    update_data = {"$set": {"settings": settings.dict()}}
    # MONGO: we are using upsert here temporarily.
    # Because the settings must exist with default settings on user creation
    user_collection.update_one({"_id": user_uid}, update_data, upsert=True)

    # if every thing is fine we return the settings in the cookies
    response.set_cookie(key="llm_name", value=settings.llm_name, samesite="None", secure=True)
    logger.info("LLM settings updated")
    return {"message": "LLM settings saved"}


@router.get("/get-llm-settings")
async def get_llm_settings(request: Request, api_key: Annotated[str | None, Cookie()] = None):
    user_uid = request.state.verified_user["user_id"]

    user_settings = user_collection.find_one({"_id": user_uid}).get("settings")
    logger.info(f"LLM settings retrieved: {user_settings}")
    if user_settings is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    logger.info(f"LLM settings retrieved: {user_settings}")
    logger.info("Appending api_key to response from cookie")
    user_settings["api_key"] = api_key
    return user_settings
