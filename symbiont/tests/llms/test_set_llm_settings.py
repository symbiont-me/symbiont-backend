# TODO fix this test
# import pytest
# from fastapi.testclient import TestClient
# from symbiont.routers.llm_settings import router
# from symbiont.mongodb import users_collection
# import mongomock
# from fastapi import FastAPI, Request

# # Create a FastAPI app and include the router
# app = FastAPI()
# app.include_router(router)

# # Use mongomock to mock the MongoDB collection
# users_collection = mongomock.MongoClient().db.collection

# # Create a test client
# client = TestClient(app)


# @pytest.fixture
# def mock_user():
#     user = {"_id": "test_user_id", "settings": {}}
#     users_collection.insert_one(user)
#     return user


# @pytest.fixture
# def mock_request_state(monkeypatch):
#     class MockState:
#         verified_user = {"user_id": "test_user_id"}

#     def mock_request(*args, **kwargs):
#         request = Request(scope={"type": "http"})
#         request.state = MockState()
#         return request

#     monkeypatch.setattr("symbiont.routers.llm_settings.Request", mock_request)


# def test_set_llm_settings(mock_user, mock_request_state):
#     response = client.post("/set-llm-settings", json={"llm_name": "test_llm", "api_key": "test_api_key"})
#     assert response.status_code == 200
#     assert response.json() == {"message": "LLM settings saved"}

#     # Check if the settings were updated in the mock database
#     updated_user = users_collection.find_one({"_id": "test_user_id"})
#     assert updated_user["settings"]["llm_name"] == "test_llm"
#     assert "api_key" not in updated_user["settings"]

#     # Check if the cookies were set correctly
#     assert response.cookies["api_key"] == "test_api_key"
#     assert response.cookies["llm_name"] == "test_llm"
