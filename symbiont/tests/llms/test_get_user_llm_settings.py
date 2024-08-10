import pytest
from unittest import mock
from symbiont.llms import get_user_llm_settings
from fastapi import HTTPException


@pytest.fixture
def mock_mongo():
    with mock.patch("symbiont.llms.users_collection") as mock_collection:
        yield mock_collection


def test_get_user_llm_settings(mock_mongo):
    """
    Test function for getting user language model settings using the provided mock mongo object.
    """
    user_uid = "user123"
    settings = {"language_model": "GPT-3", "response_length": 150, "temperature": 0.7}
    mock_mongo.find_one.return_value = {"settings": settings}

    result = get_user_llm_settings(user_uid)

    assert result == settings
    mock_mongo.find_one.assert_called_once_with({"_id": user_uid})


def test_get_user_llm_settings_user_not_found(mock_mongo):
    """
    Tests the get_user_llm_settings function when the user is not found in the database.

    This test case checks that the function raises an HTTPException with a 404 status code and a "User not found" detail message.

    Parameters:
    mock_mongo: A mock object representing the MongoDB collection.

    Returns:
    None
    """
    user_uid = "user_not_exist"
    mock_mongo.find_one.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        get_user_llm_settings(user_uid)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "User not found"
    mock_mongo.find_one.assert_called_once_with({"_id": user_uid})


if __name__ == "__main__":
    pytest.main()
