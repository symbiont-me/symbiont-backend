from fastapi import APIRouter, Request, HTTPException, Cookie
from fastapi.responses import JSONResponse
from ..models import TextUpdateRequest
from ..mongodb import studies_collection
from symbiont.mongodb.utils import user_exists, check_user_authorization
from symbiont.llms import init_llm, get_user_llm_settings
from pydantic import BaseModel, ValidationError
from typing import Annotated, Optional, Any

from langchain import PromptTemplate
from langchain_core.messages.ai import AIMessage
from .. import logger
from langchain.chains import LLMChain

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


class CompletionRequest(BaseModel):
    sentence_start: str


@router.post("/completion")
async def completion(
    completion_request: CompletionRequest,
    request: Request,
    api_key: Annotated[str | None, Cookie()] = None,
):
    if not completion_request.sentence_start or not api_key:
        return JSONResponse(
            {"response": ""},
        )

    session_data = {
        "user_id": request.state.session.get_user_id(),
    }

    user_uid = session_data["user_id"]
    await user_exists(user_uid)
    llm_settings = get_user_llm_settings(user_uid)
    logger.debug(f"Initializing {llm_settings=}")
    llm = init_llm(llm_settings, api_key)
    if llm is None:
        raise HTTPException(status_code=404, detail="No LLM found!")

    template = "Complete the following sentence: {sentence_start}"
    prompt = PromptTemplate(input_variables=["sentence_start"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    llm_response = chain.run(sentence_start=completion_request.sentence_start)

    return JSONResponse({"response": llm_response})
