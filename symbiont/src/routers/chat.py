from requests import api
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from ..models import ChatRequest, ChatMessage, LLMModel

from ..pinecone.pc import get_chat_context
from langchain.prompts import PromptTemplate
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import AsyncGenerator
from google.cloud.firestore import ArrayUnion
from ..utils.db_utils import get_document_ref, get_document_dict
from ..utils.llm_utils import truncate_prompt
from pydantic import BaseModel
from langchain.chains import LLMChain
from pydantic import BaseModel

from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI

from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from ..llms import (
    create_user_prompt,
    generate_anthropic_response,
    isOpenAImodel,
    isAnthropicModel,
    get_user_llm_settings,
    generate_openai_response,
)

####################################################
#                   CHAT                           #
####################################################

router = APIRouter()


def get_combined_chat_context(study_id: str, user_uid: str, user_query: str):
    all_resource_identifiers = []
    # # get all resources for the study
    # TODO use a function that does not require user_uid to be passed
    study_dict = get_document_dict("studies_", "userId", user_uid, study_id)
    if study_dict is None:
        raise HTTPException(status_code=404, detail="No such document!")
    resources = study_dict.get("resources", [])
    if resources is None:
        raise HTTPException(status_code=404, detail="No Resources Found")
    # get the identifier for each resource
    all_resource_identifiers = [resource.get("identifier") for resource in resources]
    # # get the context for each resource
    contexts = [
        get_chat_context(user_query, resource_identifier, 1)
        for resource_identifier in all_resource_identifiers
    ]
    # # TODO keep the context within the model's max token limit
    return " ".join(contexts)


@router.post("/chat")
async def chat(chat: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Handles the chat endpoint.

    Args:
        chat (ChatRequest): The chat request object containing user input.
        request (Request): The request object.
        background_tasks (BackgroundTasks): The background tasks object.

    Returns:
        StreamingResponse: The response containing the generated chat message from the language model.
    """
    user_uid = request.state.verified_user["user_id"]
    user_query = chat.user_query
    previous_message = ""  # TODO remove this feature as previous_message makes makes the context poorer
    study_id = chat.study_id
    resource_identifier = chat.resource_identifier
    background_tasks.add_task(
        save_chat_message_to_db,
        chat_message=user_query,
        studyId=study_id,
        role="user",
        user_uid=user_uid,
    )

    context = ""

    print(chat.combined, "COMBINED")
    print(resource_identifier, "RESOURCE IDENTIFIER")

    if not chat.combined and resource_identifier is None:
        raise HTTPException(  # TODO this should be a 400
            status_code=404, detail="Resource Identifier Required"
        )
    if chat.combined:
        print("GETTING COMBINED CONTEXT")
        context = get_combined_chat_context(chat.study_id, user_uid, chat.user_query)
    if not chat.combined and resource_identifier is not None:
        print("GETTING SINGLE CONTEXT")
        context = get_chat_context(user_query, resource_identifier)

    context = truncate_prompt(context)

    # print("CONTEXT", context)

    llm = get_user_llm_settings(user_uid)
    if llm is None:
        raise HTTPException(status_code=404, detail="No LLM settings found!")
    prompt = create_user_prompt(user_query, context, previous_message)
    print(llm, "LLM")

    # NOTE a bit slow
    # TODO fix streaming response
    async def generate_llm_response() -> AsyncGenerator[str, None]:
        """
        This asynchronous generator function streams the response from the language model (LLM) in chunks.
        For each chunk received from the LLM, it appends the chunk to the 'llm_response' string and yields
        the chunk to the caller. After all chunks have been received and yielded, it schedules a background task
        to save the complete response to the database as a chat message from the 'bot' role.
        """

        llm_response = ""

        if isOpenAImodel(llm["llm_name"]):
            async for chunk in generate_openai_response(
                model=llm["llm_name"],
                api_key=llm["api_key"],
                max_tokens=1500,
                user_query=user_query,
                context=context,
            ):
                llm_response += chunk
                yield chunk

            background_tasks.add_task(
                save_chat_message_to_db,
                chat_message=llm_response,
                studyId=study_id,
                role="bot",
                user_uid=user_uid,
            )

        elif isAnthropicModel(
            llm["llm_name"]
        ):  # Changed to elif to avoid overlapping if conditions

            async for chunk in generate_anthropic_response(
                model=llm["llm_name"],
                max_tokens=1500,
                api_key=llm["api_key"],
                user_query=user_query,
                context=context,
                previous_message=previous_message,
            ):
                llm_response += chunk
                yield chunk

            background_tasks.add_task(
                save_chat_message_to_db,
                chat_message=llm_response,
                studyId=study_id,
                role="bot",
                user_uid=user_uid,
            )

    return StreamingResponse(generate_llm_response())


@router.get("/get-chat-messages")
async def get_chat_messages(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_data = get_document_dict("studies_", "userId", user_uid, studyId)
    if study_data is None:
        raise HTTPException(status_code=404, detail="No such document!")
    if "chatMessages" in study_data:
        return {"chatMessages": study_data["chatMessages"]}


@router.delete("/delete-chat-messages")
async def delete_chat_messages(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    print("DELETING CHAT MESSAGES")
    doc_ref = get_document_ref("studies_", "userId", user_uid, studyId)
    if doc_ref is None:
        raise HTTPException(status_code=404, detail="No such document!")

    doc_ref.update({"chatMessages": []})
    return {"message": "Chat messages deleted!", "status_code": 200}


def save_chat_message_to_db(chat_message: str, studyId: str, role: str, user_uid: str):

    doc_ref = get_document_ref("studies_", "userId", user_uid, studyId)
    if doc_ref is None:
        raise HTTPException(status_code=404, detail="No such document!")
    new_chat_message = ChatMessage(
        role=role, content=chat_message, createdAt=datetime.now()
    ).model_dump()
    doc_ref.update({"chatMessages": ArrayUnion([new_chat_message])})
