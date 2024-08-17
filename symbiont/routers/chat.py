from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Cookie
from ..models import ChatRequest, ChatMessage, Citation

from fastapi.responses import StreamingResponse
import datetime
from typing import AsyncGenerator


from ..llms import (
    get_user_llm_settings,
    init_llm,
    get_llm_response,
)
from ..pinecone.pc import PineconeService
from .. import logger
import time
from typing import Annotated, List
import asyncio
from ..mongodb import studies_collection

from symbiont.vector_dbs.chat_context_service import ChatContextService


####################################################
#                   CHAT                           #
####################################################

router = APIRouter()


async def return_no_context_response(response: str = "") -> AsyncGenerator[str, None]:
    gen = iter(response.split())
    for chunk in gen:
        response += chunk + " "
        await asyncio.sleep(0.05)
        yield chunk + " "


# TODO wrap in try except
@router.post("/chat")
async def chat(
    chat: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: Annotated[str | None, Cookie()] = None,
):
    s = time.time()
    user_uid = request.state.verified_user["user_id"]

    if api_key is None:
        raise HTTPException(status_code=404, detail="No API key found!")

    user_query = chat.user_query
    #### INIT LLM ####
    llm_settings = get_user_llm_settings(user_uid)
    logger.debug(f"Initializing {llm_settings=}")
    llm = init_llm(llm_settings, api_key)

    study_id = chat.study_id
    resource_identifier = chat.resource_identifier
    # TODO batch the write operations
    save_chat_message_to_db(
        chat_message=user_query,
        studyId=study_id,
        role="user",
        user_uid=user_uid,
    )
    logger.info(resource_identifier)

    context = ""
    citations = []
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

    chat_context_service = ChatContextService(study_id=study_id, resource_identifier=resource_identifier)

    if chat.combined:
        logger.info("GETTING CONTEXT FOR COMBINED RESOURCES")
        chat_context_results = chat_context_service.get_combined_chat_context(user_query)
        if chat_context_results is None:
            logger.debug("No context found, retuning no context response")
            no_context_response = (
                "I am sorry, there is no information available in the documents to answer your question."
            )
            save_chat_message_to_db(
                chat_message=no_context_response,
                citations=citations,
                studyId=study_id,
                role="bot",
                user_uid=user_uid,
            )
            return StreamingResponse(
                return_no_context_response(no_context_response),
                media_type="text/event-stream",
            )
        context = chat_context_results[0]
        citations = chat_context_results[1]

    if not chat.combined:
        logger.info("GETTING CONTEXT FOR A SINGLE RESOURCE")
        context_start_time = time.time()

        result = chat_context_service.get_single_chat_context(user_query)

        # result = await pc_service.get_single_chat_context()
        if result is None:
            logger.debug("No context found, retuning no context response")
            no_context_response = (
                "I am sorry, there is no information available in the documents to answer your question."
            )
            save_chat_message_to_db(
                chat_message=no_context_response,
                citations=citations,
                studyId=study_id,
                role="bot",
                user_uid=user_uid,
            )
            return StreamingResponse(
                return_no_context_response(no_context_response),
                media_type="text/event-stream",
            )

        context = result[0]
        citations = result[1]
        context_elapsed_time = time.time() - context_start_time
        logger.debug(f"fetched context in {str(datetime.timedelta(seconds=context_elapsed_time))}")

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
                context=context,  # TODO fix type issue
            ):
                llm_response += chunk
                yield chunk
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

        save_chat_message_to_db(
            chat_message=llm_response,
            citations=citations,
            studyId=study_id,
            role="bot",
            user_uid=user_uid,
        )

    elasped_time = time.time() - s
    logger.info(f"It took {elasped_time} to start the chat ")
    return StreamingResponse(generate_llm_response(), media_type="text/event-stream")

    #
    #
    #
    #


@router.get("/get-chat-messages")
async def get_chat_messages(studyId: str):
    logger.debug("LOADING CHATS")
    chats = studies_collection.find_one({"_id": studyId})["chat"]
    logger.debug(chats)
    return {"chat": chats}


@router.delete("/delete-chat-messages")
async def delete_chat_messages(studyId: str):
    studies_collection.update_one({"_id": studyId}, {"$set": {"chat": []}})
    logger.info("Chat messages deleted!")
    return {"message": "Chat messages deleted!", "status_code": 200}


def save_chat_message_to_db(
    chat_message: str,
    studyId: str,
    role: str,
    user_uid: str,
    citations: List[Citation] = [],
):
    new_chat_message = ChatMessage(
        role=role,
        content=chat_message,
        citations=citations,
        createdAt=datetime.datetime.now(),
    ).model_dump()
    studies_collection.find_one_and_update({"_id": studyId}, {"$push": {"chat": new_chat_message}})
    if role == "bot":
        logger.info("Bot message saved to db")
    else:
        logger.info("User message saved to db")
    return {"message": "Chat message saved to db", "status_code": 200}
