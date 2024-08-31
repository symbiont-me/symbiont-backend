import pytest
from fastapi import HTTPException
from symbiont.llms import get_user_llm_settings
import mongomock
from unittest.mock import patch


@pytest.fixture
def mock_mongo():
    mock_db = mongomock.MongoClient().db
    with patch("symbiont.llms.users_collection", mock_db.users_collection):
        yield mock_db.users_collection


def test_get_user_llm_settings(mock_mongo):
    """
    Test function for getting user language model settings using the provided mock mongo object.
    """
    user_uid = "user123"
    settings = {"language_model": "GPT-3", "response_length": 150, "temperature": 0.7}

    # Inserting a mock user document
    mock_mongo.insert_one({"_id": user_uid, "settings": settings})

    result = get_user_llm_settings(user_uid)

    assert result == settings

    # Optionally check the internal state of the mock database
    assert mock_mongo.find_one({"_id": user_uid}) == {"_id": user_uid, "settings": settings}


def test_get_user_llm_settings_user_not_found(mock_mongo):
    """
    Tests the get_user_llm_settings function when the user is not found in the database.

    This test case checks that the function raises an HTTPException with a 404 status code and a "User not found"
    detail message.

    Parameters:
    mock_mongo: A mock object representing the MongoDB collection.

    Returns:
    None
    """
    user_uid = "user_not_exist"

    with pytest.raises(HTTPException) as exc_info:
        get_user_llm_settings(user_uid)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "User not found"

    # Verify that the collection was queried
    assert mock_mongo.find_one({"_id": user_uid}) is None


if __name__ == "__main__":
    pytest.main()
