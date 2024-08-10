import pytest
from fastapi.testclient import TestClient
from symbiont.routers.chat import save_chat_message_to_db, router
from symbiont.models import ChatMessage, Citation
from fastapi import FastAPI
from unittest.mock import patch
import datetime

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_studies_collection():
    with patch("symbiont.routers.chat.studies_collection") as mock:
        yield mock


@pytest.fixture
def mock_datetime():
    with patch("symbiont.routers.chat.datetime") as mock:
        mock.datetime.now.return_value = datetime.datetime(2023, 1, 1)
        yield mock


"""
    Test the save_chat_message_to_db function.

    This function tests the save_chat_message_to_db function by mocking the studies_collection and mock_datetime objects.
    It creates a chat message, studyId, role, user_uid, and citations, and then calls the save_chat_message_to_db function with these parameters.
    The function asserts that the response is equal to {"message": "Chat message saved to db", "status_code": 200}.
    It also asserts that the mock_studies_collection.find_one_and_update method is called once with the correct arguments.

    Parameters:
    - mock_studies_collection (MagicMock): A mock object representing the studies_collection.
    - mock_datetime (MagicMock): A mock object representing the datetime.

    Returns:
    None
"""


def test_save_chat_message_to_db(mock_studies_collection, mock_datetime):
    chat_message = "Hello, this is a test message."
    studyId = "test_study_id"
    role = "user"
    user_uid = "test_user_uid"
    citations = [Citation(source="test_source", text="test_text", page=1)]

    response = save_chat_message_to_db(chat_message, studyId, role, user_uid, citations)

    new_chat_message = ChatMessage(
        role=role, content=chat_message, citations=citations, createdAt=datetime.datetime(2023, 1, 1)
    ).model_dump()

    mock_studies_collection.find_one_and_update.assert_called_once_with(
        {"_id": studyId}, {"$push": {"chat": new_chat_message}}
    )

    assert response == {"message": "Chat message saved to db", "status_code": 200}
