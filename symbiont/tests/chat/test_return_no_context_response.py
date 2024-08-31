import pytest
from symbiont.routers.chat import return_no_context_response


@pytest.mark.asyncio
async def test_return_no_context_response():
    response = "This is a test response"
    expected_chunks = ["This ", "is ", "a ", "test ", "response "]

    gen = return_no_context_response(response)
    chunks = [chunk async for chunk in gen]

    assert chunks == expected_chunks
