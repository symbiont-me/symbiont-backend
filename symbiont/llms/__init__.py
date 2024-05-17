import re
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import ValidationError
from pydantic import BaseModel
from firebase_admin import firestore
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import HTTPException
import time
import datetime
from .. import logger
import os
from ..mongodb import users_collection


google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")


def create_prompt(user_query: str, context: str):
    prompt_template = PromptTemplate.from_template(
        """
        You are a well-informed AI assistant. 
        The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
        AI has the sum of all knowledge in their brain, and is able to accurately answer nearly 
        any question about any topic in conversation.
        AI assistant will take into account any information that is provided and construct 
        a reasonable and well thought response.
        START OF INFORMATION {context} END OF INFORMATION
        If it is not enought to provide a reasonable answer to the question, the AI assistant will say:
        "I'm sorry, but I don't know the answer to that question. But my educated opinion would..."
        AI assistant will not invent anything and do its best to provide accurate information.
        Output Format: Return your answer in valid {output_format} format
        {user_query} 
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
    except ValidationError as e:
        if e.errors():
            logger.error(f"Error initializing LLM: {e.errors()}")
            raise HTTPException(status_code=400, detail="Check your API key and LLM settings")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise HTTPException(status_code=400, detail="Error initializing LLM")


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
    logger.debug(
        f"Generated {num_chunks} chunks in {str(datetime.timedelta(seconds=llm_elapsed_time))}"
        f"at a speed of {round(speed,2)} chunk/s."
    )

    # system = system_prompt.split("Question:")[0]  # Extract system part from the prompt
    # human = user_query

    # prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # chain = prompt | llm
    # for chunk in chain.stream({"system": SystemMessage, "human": HumanMessage}):
    #     yield chunk.content


# TODO move to routers/llm_settings.py
def get_user_llm_settings(user_uid: str):
    user_llm_settings = users_collection.find_one({"_id": user_uid}).get("settings")
    logger.debug(f"User LLM Settings: {user_llm_settings}")
    return user_llm_settings
