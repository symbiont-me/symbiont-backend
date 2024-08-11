import pytest
from fastapi.testclient import TestClient
from symbiont.routers.chat import router
from symbiont.mongodb import studies_collection
from unittest.mock import patch
import mongomock

client = TestClient(router)

"""
    A fixture used to mock the studies_collection in the chat router.

    Yields a mock object representing the studies_collection.
"""


@pytest.fixture
def mock_studies_collection():
    with patch("symbiont.routers.chat.studies_collection", mongomock.MongoClient().db.collection) as mock:
        yield mock


"""
    Tests the get_chat_messages endpoint by mocking the studies_collection.

    Args:
    mock_studies_collection: A mock object representing the studies_collection.

    Returns:
    None
"""


def test_get_chat_messages(mock_studies_collection):
    study_id = "test_study_id"
    mock_chats = [{"role": "user", "content": "Hello"}]
    mock_studies_collection.insert_one({"_id": study_id, "chat": mock_chats})

    response = client.get(f"/get-chat-messages?studyId={study_id}")

    assert response.status_code == 200
    assert response.json() == {"chat": mock_chats}
    assert mock_studies_collection.find_one({"_id": study_id}) is not None
