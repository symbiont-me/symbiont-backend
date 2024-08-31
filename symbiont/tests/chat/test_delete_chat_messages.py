import pytest
from fastapi.testclient import TestClient
from symbiont.routers.chat import router
from symbiont.mongodb import studies_collection
import mongomock
from unittest.mock import patch

client = TestClient(router)


@pytest.fixture
def mock_study():
    study_id = "test_study_id"
    with patch("symbiont.mongodb.studies_collection", mongomock.MongoClient().db.collection):
        studies_collection.insert_one({"_id": study_id, "chat": ["test message"]})
        yield study_id
        studies_collection.delete_one({"_id": study_id})


def test_delete_chat_messages(mock_study):
    with patch("symbiont.mongodb.studies_collection", mongomock.MongoClient().db.collection):
        response = client.delete(f"/delete-chat-messages?studyId={mock_study}")
        assert response.status_code == 200
        assert response.json() == {"message": "Chat messages deleted!", "status_code": 200}

        study = studies_collection.find_one({"_id": mock_study})
        assert study is not None
        assert study["chat"] == []
