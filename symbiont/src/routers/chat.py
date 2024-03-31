from requests import api
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from ..models import ChatRequest, ChatMessage, LLMModel


from langchain.prompts import PromptTemplate
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import AsyncGenerator
from google.cloud.firestore import ArrayUnion
from ..utils.db_utils import StudyService
from ..utils.llm_utils import truncate_prompt
from pydantic import BaseModel
from langchain.chains import LLMChain
from pydantic import BaseModel

from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI

from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from ..llms import (
    get_user_llm_settings,
    init_llm,
    get_llm_response,
)
from ..pinecone.pc import PineconeService
from .. import logger

####################################################
#                   CHAT                           #
####################################################

router = APIRouter()


@router.post("/chat")
async def chat(chat: ChatRequest, request: Request, background_tasks: BackgroundTasks):

    user_uid = request.state.verified_user["user_id"]
    user_query = chat.user_query
    #### INIT LLM ####
    llm_settings = get_user_llm_settings(user_uid)
    llm = init_llm(llm_settings)
    # TODO remove after testing, shouldn't be needing this as error is handled in init_llm
    if llm is None:
        raise HTTPException(status_code=404, detail="No LLM settings found!")

    # TODO remove this feature as previous_message makes makes the context poor
    previous_message = ""
    study_id = chat.study_id
    resource_identifier = chat.resource_identifier
    background_tasks.add_task(
        save_chat_message_to_db,
        chat_message=user_query,
        studyId=study_id,
        role="user",
        user_uid=user_uid,
    )

    logger.info(resource_identifier)

    context = ""
    context_metadata = []

    if not chat.combined and resource_identifier is None:
        raise HTTPException(  # TODO this should be a 400
            status_code=404, detail="Please select a resource"
        )

    if resource_identifier is None:
        raise HTTPException(  # TODO this should be a 400
            status_code=404, detail="Please select a resource"
        )

    pc_service = PineconeService(
        study_id=study_id,
        resource_identifier=resource_identifier,
        user_uid=user_uid,
        user_query=user_query,
    )
    if chat.combined:
        logger.info("GETTING CONTEXT FOR COMBINED RESOURCES")
        context = await pc_service.get_combined_chat_context()
    if not chat.combined:
        logger.info("GETTING CONTEXT FOR A SINGLE RESOURCE")
        context = await pc_service.get_single_chat_context()

    # if context == "":
    #     response = "I am sorry, there is no information available in the documents to answer your question."

    async def generate_llm_response() -> AsyncGenerator[str, None]:
        """
        This asynchronous generator function streams the response from the language model (LLM) in chunks.
        For each chunk received from the LLM, it appends the chunk to the 'llm_response' string and yields
        the chunk to the caller. After all chunks have been received and yielded, it schedules a background task
        to save the complete response to the database as a chat message from the 'bot' role.
        """
        try:
            llm_response = ""
            async for chunk in get_llm_response(
                llm=llm,
                user_query=user_query,
                context=context,  # TODO needs to be fixed
            ):
                llm_response += chunk
                yield chunk

            # TODO fix this
            if context == "":
                llm_response = "I am sorry, there is no information available in the documents to answer your question."
                yield llm_response
                background_tasks.add_task(
                    save_chat_message_to_db,
                    chat_message=llm_response,
                    studyId=study_id,
                    role="bot",
                    user_uid=user_uid,
                )
            background_tasks.add_task(
                save_chat_message_to_db,
                chat_message=llm_response,
                studyId=study_id,
                role="bot",
                user_uid=user_uid,
            )
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(generate_llm_response())


@router.get("/get-chat-messages")
async def get_chat_messages(studyId: str, request: Request):
    study_service = StudyService(request.state.verified_user["user_id"], studyId)
    study_data = study_service.get_document_dict()
    if study_data is None:
        raise HTTPException(status_code=404, detail="No such document!")
    if "chatMessages" in study_data:
        return {"chatMessages": study_data["chatMessages"]}


@router.delete("/delete-chat-messages")
async def delete_chat_messages(studyId: str, request: Request):
    study_service = StudyService(request.state.verified_user["user_id"], studyId)
    print("DELETING CHAT MESSAGES")
    doc_ref = study_service.get_document_ref()
    if doc_ref is None:
        raise HTTPException(status_code=404, detail="No such document!")

    doc_ref.update({"chatMessages": []})
    return {"message": "Chat messages deleted!", "status_code": 200}


def save_chat_message_to_db(chat_message: str, studyId: str, role: str, user_uid: str):
    # TODO improve this
    study_service = StudyService(user_uid, studyId)
    doc_ref = study_service.get_document_ref()
    if doc_ref is None:
        raise HTTPException(status_code=404, detail="No such document!")
    new_chat_message = ChatMessage(
        role=role, content=chat_message, createdAt=datetime.now()
    ).model_dump()
    doc_ref.update({"chatMessages": ArrayUnion([new_chat_message])})
