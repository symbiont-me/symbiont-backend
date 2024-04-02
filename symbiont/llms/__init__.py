import re
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from firebase_admin import firestore
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import HTTPException
from pydantic import SecretStr
import time
import datetime
from .. import logger


def create_prompt(user_query: str, context: str):
    prompt_template = PromptTemplate.from_template(
        """
        You are a well-informed AI assistant. 
        The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
        AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation.
        AI assistant will take into account any DOCUMENT BLOCK that is provided in a conversation.
        START DOCUMENT BLOCK {context} END OF DOCUMENT BLOCK
        If the context does not provide the answer to the question or the context is empty, the AI assistant will say:
        "I'm sorry, but I don't know the answer to that question."
        AI assistant will not invent anything that is not drawn directly from the context.
        AI will be as detailed as possible.
        Output Format: Return your answer in valid {output_format} Format
        Question: {user_query}
        Answer: 
    """
    )

    prompt = prompt_template.format(
        context=context,
        output_format="Markdown",
        user_query=user_query,
    )
    return prompt


# TODO move to utils
def isOpenAImodel(llm_name: str) -> bool:
    return bool(re.match(r"gpt", llm_name))


def isAnthropicModel(llm_name: str) -> bool:
    return bool(re.match(r"claude", llm_name))


def isGoogleModel(llm_name: str) -> bool:
    return bool(re.match(r"gemini", llm_name))


class UsersLLMSettings(BaseModel):
    llm_name: str
    # api_key: SecretStr
    # max_tokens: int
    # temperature: float


def init_llm(settings: UsersLLMSettings, api_key: str):
    if settings is None:
        raise HTTPException(status_code=400, detail="Please set LLM Settings")
    if api_key is None:
        raise HTTPException(status_code=400, detail="Please provide an API key")
    logger.debug(f"Initializing LLM with settings: {settings}")
    try:
        llm = None
        if isOpenAImodel(settings["llm_name"]):
            llm = ChatOpenAI(
                model=settings["llm_name"],
                api_key=api_key,
                max_tokens=1500,
                temperature=0,
            )
            return llm
        elif isAnthropicModel(settings["llm_name"]):
            llm = ChatAnthropic(
                model_name=settings["llm_name"],
                anthropic_api_key=api_key,
                temperature=0,
            )
            return llm
        elif isGoogleModel(settings["llm_name"]):
            llm = ChatGoogleGenerativeAI(
                model=settings["llm_name"],
                google_api_key=api_key,
                max_tokens=1500,
                temperature=0,
                convert_system_message_to_human=True,
            )
            return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")


async def get_llm_response(llm, user_query: str, context: str):
    prompt = create_prompt(user_query, context)
    num_chunks = 0
    llm_start_time = time.time()
    for chunk in llm.stream(prompt):
        if num_chunks == 0:
                    time_to_first_token = time.time() - llm_start_time
                    logger.info(f"Time to first token (TTFT) {str(datetime.timedelta(seconds=time_to_first_token))}")
        num_chunks += 1
        yield chunk.content
    llm_elapsed_time = time.time() - llm_start_time
    speed = num_chunks / llm_elapsed_time
    logger.debug(f"Generated {num_chunks} chunks in {str(datetime.timedelta(seconds=llm_elapsed_time))} at a spped of {round(speed,2)} chunk/s.")

    # system = system_prompt.split("Question:")[0]  # Extract system part from the prompt
    # human = user_query

    # prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # chain = prompt | llm
    # for chunk in chain.stream({"system": SystemMessage, "human": HumanMessage}):
    #     yield chunk.content


# TODO move to routers/llm_settings.py
def get_user_llm_settings(user_uid: str):
    db = firestore.client()
    doc_ref = db.collection("users").document(user_uid)
    if not doc_ref:
        raise ValueError("User not found")
    else:
        settings = doc_ref.get().get("settings")
        logger.info(f"User settings: {settings}")
        return settings
