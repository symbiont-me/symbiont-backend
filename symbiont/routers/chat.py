from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from ..models import ChatRequest, ChatMessage, LLMModel
from firebase_admin import firestore

from fastapi.responses import StreamingResponse
import datetime
from typing import AsyncGenerator
from google.cloud.firestore import ArrayUnion
from ..utils.db_utils import StudyService


from ..llms import (
    get_user_llm_settings,
    init_llm,
    get_llm_response,
)
from ..pinecone.pc import PineconeService
from .. import logger
import time

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
    # TODO review these conditions
    if not chat.combined and resource_identifier is None:
        raise HTTPException(status_code=404, detail="Please select a resource")

    if resource_identifier is None:
        raise HTTPException(status_code=404, detail="Please select a resource")

    logger.debug("Initializing pc service")
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
        context_start_time = time.time()
        context = await pc_service.get_single_chat_context()
        context_elapsed_time = time.time() - context_start_time
        logger.debug(
            f"fetched context in {str(datetime.timedelta(seconds=context_elapsed_time))}"
        )

    no_context_response = ""
    if not context:
        logger.debug("No context found, retuning no context response")

        def return_no_context_response():
            nonlocal no_context_response
            response = "I am sorry, there is no information available in the documents to answer your question."
            gen = iter(response.split())
            for chunk in gen:
                no_context_response += chunk + " "
                time.sleep(0.05)
                yield chunk
            logger.debug("Adding bg task")

        background_tasks.add_task(
            save_chat_message_to_db,
            chat_message=no_context_response,
            studyId=study_id,
            role="bot",
            user_uid=user_uid,
        )
        logger.debug("Returning response")
        return StreamingResponse(
            return_no_context_response(), media_type="text/event-stream"
        )

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
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

        background_tasks.add_task(
            save_chat_message_to_db,
            chat_message=llm_response,
            studyId=study_id,
            role="bot",
            user_uid=user_uid,
        )

    return StreamingResponse(generate_llm_response(), media_type="text/event-stream")


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
    db = firestore.client()
    doc_ref = db.collection("studies").document(studyId)
    if doc_ref.get().to_dict() is None:
        raise HTTPException(status_code=404, detail="No such document!")
    new_chat_message = ChatMessage(
        role=role, content=chat_message, createdAt=datetime.datetime.now()
    ).model_dump()
    doc_ref.update({"chatMessages": ArrayUnion([new_chat_message])})
    if role == "bot":
        logger.info("Bot message saved to db")
    else:
        logger.info("User message saved to db")
    return {"message": "Chat message saved to db", "status_code": 200}
