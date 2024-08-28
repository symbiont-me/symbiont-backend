import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from symbiont.routers.chat import generate_llm_response


@pytest.mark.asyncio
async def test_generate_llm_response():
    # Mock the dependencies
    async def mock_get_llm_response(*args, **kwargs):
        chunks = ["chunk1", "chunk2", "chunk3"]
        for chunk in chunks:
            yield chunk

    mock_save_chat_message_to_db = AsyncMock()

    # Define the inputs
    llm = "mock_llm"
    user_query = "mock_query"
    context = "mock_context"
    citations = []
    study_id = "mock_study_id"
    user_uid = "mock_user_uid"

    # Call the function
    gen = generate_llm_response(
        llm=llm,  # TODO Update type hint
        user_query=user_query,
        context=context,
        citations=citations,
        study_id=study_id,
        user_uid=user_uid,
        get_llm_response=mock_get_llm_response,
        save_chat_message_to_db=mock_save_chat_message_to_db,
    )

    # Collect the results
    result = [chunk async for chunk in gen]

    # Assertions
    assert result == ["chunk1", "chunk2", "chunk3"]
    mock_save_chat_message_to_db.assert_called_once_with(
        chat_message="chunk1chunk2chunk3",
        citations=citations,
        studyId=study_id,
        role="bot",
        user_uid=user_uid,
    )
