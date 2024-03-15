import os
from ..models import LLMModel
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_openai import OpenAI


def create_user_prompt(user_query: str, context: str, previous_message: str | None):
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


# TODO move to own file
async def generate_anthropic_response(
    model: LLMModel,
    max_tokens: int,
    user_query: str,
    context: str,
    previous_message: str | None,
):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key is None:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    chat = ChatAnthropic(
        temperature=0, anthropic_api_key=api_key, model_name=LLMModel.CLAUDE_INSTANT_1_2
    )

    system_prompt = create_user_prompt(user_query, context, previous_message)
    system = system_prompt.split("Question:")[0]  # Extract system part from the prompt
    human = user_query
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    for chunk in chain.stream({}):
        print(chunk.content)
        yield chunk.content


# TODO move to models/llm_settings.py
class DefaultLLMSettings(BaseModel):
    model: str = LLMModel.GPT_3_5_TURBO_INSTRUCT
    max_tokens: int = 1500
    temperature: float = 0.75


# TODO move to routers/llm_settings.py
def get_user_llm_settings():
    # TODO get user settings from the database
    # if there are none use the default settings
    default_settings = DefaultLLMSettings()
    return OpenAI(
        model=default_settings.model,
        temperature=default_settings.temperature,
    )
