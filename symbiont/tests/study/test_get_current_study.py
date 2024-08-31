# TODO fix this
# import pytest
# from fastapi.testclient import TestClient
# from symbiont.routers.study import router
# import mongomock
# from fastapi import FastAPI

# # Create a FastAPI app and include the router
# app = FastAPI()
# app.include_router(router)

# # Use mongomock to mock the MongoDB collections
# studies_collection = mongomock.MongoClient().db.collection

# # Create a test client
# client = TestClient(app)


# @pytest.fixture
# def mock_study():
#     study = {
#         "_id": "test_study_id",
#         "userId": "test_user_id",
#         "name": "Test Study",
#         "description": "A study for testing",
#     }
#     studies_collection.insert_one(study)
#     return study


# @pytest.fixture
# def mock_request():
#     class MockRequest:
#         state = type("obj", (object,), {"verified_user": {"user_id": "test_user_id"}})()

#     return MockRequest()


# def test_get_current_study(mock_study, mock_request):
#     response = client.get(
#         "/get-current-study", params={"studyId": "test_study_id"}, headers={"X-User-Id": "test_user_id"}
#     )
#     assert response.status_code == 200
#     assert response.json()["status_code"] == 200
#     assert response.json()["studies"][0]["_id"] == "test_study_id"


# def test_get_current_study_unauthorized(mock_study, mock_request):
#     mock_request.state.verified_user["user_id"] = "unauthorized_user_id"
#     response = client.get(
#         "/get-current-study", params={"studyId": "test_study_id"}, headers={"X-User-Id": "unauthorized_user_id"}
#     )
#     assert response.status_code == 404
#     assert response.json()["detail"] == "User Not Authorized to Access Study"
