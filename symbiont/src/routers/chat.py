from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from ..models import ChatRequest, ChatMessage, LLMModel
from langchain_openai import OpenAI
from ..pinecone.pc import get_chat_context
from langchain.prompts import PromptTemplate
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import AsyncGenerator
from google.cloud.firestore import ArrayUnion
from ..utils.db_utils import get_document_ref, get_document_dict
from pydantic import BaseModel
from langchain.chains import LLMChain
from typing import Optional
from typing import Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


####################################################
#                   CHAT                           #
####################################################

router = APIRouter()


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
    previous_message = chat.previous_message
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

    if resource_identifier:
        context = get_chat_context(user_query, resource_identifier)
    # TODO make a prompt function
    llm = OpenAI(
        model=LLMModel.GPT_3_5_TURBO_INSTRUCT, temperature=0.75, max_tokens=1500
    )
    prompt_template = PromptTemplate.from_template(
        """
        You are a well-informed AI assistant. 
        The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
        AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation.
        AI assistant will take into account any DOCUMENT BLOCK that is provided in a conversation.
        START DOCUMENT BLOCK {context} END OF DOCUMENT BLOCK
        If the context does not provide the answer to the question or the context is empty, the AI assistant will say,
        I'm sorry, but I don't know the answer to that question.
        AI assistant will not invent anything that is not drawn directly from the context.
        AI will keep answers short and to the point.
    Previous Message: {previous_message}
    Question: {query}
    Output Format: Return your answer in valid {output_format} Format
    """
    )

    prompt = prompt_template.format(
        query=user_query,
        context=context,
        previous_message=previous_message,
        output_format="Markdown",
    )

    # NOTE a bit slow
    async def generate_llm_response() -> AsyncGenerator[str, None]:
        """
        This asynchronous generator function streams the response from the language model (LLM) in chunks.
        For each chunk received from the LLM, it appends the chunk to the 'llm_response' string and yields
        the chunk to the caller. After all chunks have been received and yielded, it schedules a background task
        to save the complete response to the database as a chat message from the 'bot' role.
        """
        llm_response = ""
        async for chunk in llm.astream(prompt):
            llm_response += chunk
            yield chunk

        background_tasks.add_task(
            save_chat_message_to_db,
            chat_message=llm_response,
            studyId=study_id,
            role="bot",
            user_uid=user_uid,
        )

    #
    return StreamingResponse(generate_llm_response())


@router.get("/get-chat-messages")
async def get_chat_messages(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_data = get_document_dict("studies_", "userId", user_uid, studyId)
    if study_data is None:
        raise HTTPException(status_code=404, detail="No such document!")
    if "chatMessages" in study_data:
        return {"chatMessages": study_data["chatMessages"]}


def save_chat_message_to_db(chat_message: str, studyId: str, role: str, user_uid: str):

    doc_ref = get_document_ref("studies_", "userId", user_uid, studyId)
    if doc_ref is None:
        raise HTTPException(status_code=404, detail="No such document!")
    new_chat_message = ChatMessage(
        role=role, content=chat_message, createdAt=datetime.now()
    ).model_dump()
    doc_ref.update({"chatMessages": ArrayUnion([new_chat_message])})
