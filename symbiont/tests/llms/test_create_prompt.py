from symbiont.llms import create_prompt, init_llm
import pytest
from langchain.prompts.prompt import PromptTemplate
from pydantic import ValidationError
from fastapi import HTTPException


def test_create_prompt():
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
    user_query = "What is the meaning of life?"
    context = "The meaning of life is 42."
    prompt = create_prompt(user_query, context)
    assert prompt == prompt_template.format(
        context=context,
        output_format="Markdown",
        user_query=user_query,
    )
