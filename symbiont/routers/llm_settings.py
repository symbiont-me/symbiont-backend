from fastapi import APIRouter, HTTPException, status, Cookie
from pydantic import BaseModel
from fastapi import Request, Response
from typing import Annotated
from .. import logger
from ..mongodb import users_collection

from symbiont.mongodb.utils import user_exists, check_user_authorization

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
    session_data = {
        "user_id": request.state.session.get_user_id(),
    }

    user_uid = session_data["user_id"]
    await user_exists(user_uid)
    # we attach the api_key to the response as we are not storing it in the database
    response.set_cookie(
        key="api_key", value=settings.api_key, samesite="none", secure=True
    )
    # delete the api_key from the settings object for security
    del settings.api_key
    update_data = {"$set": {"settings": settings.model_dump()}}
    # Note User settings already exits with default values
    users_collection.update_one({"_id": user_uid}, update_data)

    # if every thing is fine we return the settings in the cookies
    response.set_cookie(
        key="llm_name", value=settings.llm_name, samesite="none", secure=True
    )
    logger.info("LLM settings updated")
    return {"message": "LLM settings saved"}


@router.get("/get-llm-settings")
async def get_llm_settings(
    request: Request, api_key: Annotated[str | None, Cookie()] = None
):
    session_data = {
        "user_id": request.state.session.get_user_id(),
    }

    user_uid = session_data["user_id"]
    await user_exists(user_uid)
    user = users_collection.find_one({"_id": user_uid})
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
        )
    settings = user.get("settings")
    # @note settings can be None, e.g. if the user has just signed up
    if settings is None:
        return None
    logger.info(f"LLM settings retrieved: {settings}")
    logger.info("Appending api_key to response from cookie")
    settings["api_key"] = api_key
    return settings
