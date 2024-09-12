from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest
import mongomock
from unittest.mock import patch
import logging


# Assuming users_collection is used instead of setting up a config
users_collection = None

# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI()


# Example Router


client = TestClient(app)


@pytest.fixture
def mock_mongo():
    mock_db = mongomock.MongoClient().db
    with patch("symbiont.mongodb.users_collection", mock_db.users_collection):
        yield mock_db.users_collection


@app.middleware("http")
async def add_request_attributes(request: Request, call_next):
    request.state.verified_user = {"user_id": "valid_user_id"}
    response = await call_next(request)
    return response


# TODO fix this test
# def test_get_llm_settings_success(mock_mongo):
#     user_uid = "valid_user_id"
#     settings = {"language_model": "GPT-3", "response_length": 150, "temperature": 0.7}

#     # Insert mock user data
#     mock_mongo.insert_one({"_id": user_uid, "settings": settings})

#     response = client.get("/get-llm-settings", cookies={"api_key": "test_api_key"})

#     assert response.status_code == 200
#     assert response.json() == {
#         "language_model": "GPT-3",
#         "response_length": 150,
#         "temperature": 0.7,
#         "api_key": "test_api_key",
#     }


def test_get_llm_settings_user_not_found(mock_mongo):
    response = client.get("/get-llm-settings", cookies={"api_key": "test_api_key"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


if __name__ == "__main__":
    pytest.main()
