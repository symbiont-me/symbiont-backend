import os
from ..models import LLMModel
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_openai import OpenAI
from firebase_admin import firestore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI


def create_user_prompt(user_query: str, context: str, previous_message=""):
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
        AI will be as detailed as possible.
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
    return prompt


# TODO move to utils
def isOpenAImodel(llm_name: str) -> bool:
    return bool(re.match(r"gpt", llm_name))


def isAnthropicModel(llm_name: str) -> bool:
    return bool(re.match(r"claude", llm_name))


async def generate_openai_response(
    model: LLMModel, api_key: str, max_tokens: int, user_query: str, context: str
):
    chat = ChatOpenAI(
        model_name=model,
        temperature=0.75,
        openai_api_key=api_key,
        max_tokens=max_tokens,
    )

    system_prompt = create_user_prompt(user_query, context, "")
    system = system_prompt.split("Question:")[0]  # Extract system part from the prompt
    human = user_query
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    for chunk in chain.stream({}):

        yield chunk.content


# TODO move to own file
async def generate_anthropic_response(
    model: LLMModel,
    api_key: str,
    max_tokens: int | None,
    user_query: str,
    context: str,
    previous_message: str = "",
):
    chat = ChatAnthropic(
        temperature=0,
        anthropic_api_key=api_key,
        model_name=model,
    )

    system_prompt = create_user_prompt(user_query, context, previous_message)
    system = system_prompt.split("Question:")[0]  # Extract system part from the prompt
    human = user_query
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    for chunk in chain.stream({}):

        yield chunk.content


class LLMSettings(BaseModel):
    llm_name: str
    api_key: str


# TODO move to routers/llm_settings.py
def get_user_llm_settings(user_uid: str) -> LLMSettings | None:
    db = firestore.client()
    doc_ref = db.collection("users").document(user_uid)
    if not doc_ref:
        raise ValueError("User not found")
    else:
        settings = doc_ref.get().get("settings")
        print("SETTINGS", settings)
        return settings
