from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel


router = APIRouter()


class LLMSettingsRequest(BaseModel):
    llm_name: str
    api_key: str


@router.get("/set-llm-settings")
async def set_llm_settings(settings: LLMSettingsRequest):
    return {"message": "Set LLM settings"}
