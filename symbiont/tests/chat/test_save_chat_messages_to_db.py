import pytest
from fastapi.testclient import TestClient
from symbiont.routers.chat import save_chat_message_to_db, router
from symbiont.models import ChatMessage, Citation
from fastapi import FastAPI
from unittest.mock import patch
import datetime
import mongomock

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_studies_collection():
    mock_db = mongomock.MongoClient().db
    with patch("symbiont.routers.chat.studies_collection", mock_db.studies_collection):
        yield mock_db.studies_collection


@pytest.fixture
def mock_datetime():
    with patch("symbiont.routers.chat.datetime") as mock:
        mock.datetime.now.return_value = datetime.datetime(2023, 1, 1)
        yield mock


def test_save_chat_message_to_db(mock_studies_collection, mock_datetime):
    chat_message = "Hello, this is a test message."
    studyId = "test_study_id"
    role = "user"
    user_uid = "test_user_uid"
    citations = [Citation(source="test_source", text="test_text", page=1)]

    # Prepare the mock study document
    mock_studies_collection.insert_one({"_id": studyId, "chat": []})

    response = save_chat_message_to_db(chat_message, studyId, role, user_uid, citations)

    new_chat_message = ChatMessage(
        role=role, content=chat_message, citations=citations, createdAt=datetime.datetime(2023, 1, 1)
    ).model_dump()

    updated_study = mock_studies_collection.find_one({"_id": studyId})

    # Verifying the new chat message was added to the study's chat
    assert updated_study["chat"][-1] == new_chat_message

    # Check the response
    assert response == {"message": "Chat message saved to db", "status_code": 200}
