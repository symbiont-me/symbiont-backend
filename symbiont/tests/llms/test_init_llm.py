import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from unittest.mock import patch
from symbiont.llms import init_llm, UsersLLMSettings
from symbiont.llms import isOpenAImodel, isAnthropicModel, isGoogleModel


if __name__ == "__main__":
    pytest.main()


def create_settings(llm_name):
    return UsersLLMSettings(llm_name=llm_name)


def test_init_llm_missing_api_key():
    settings = create_settings("gpt-3")
    with pytest.raises(HTTPException) as exc_info:
        init_llm(settings, "")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Please provide an API key"


@patch("symbiont.llms.isOpenAImodel", return_value=True)
@patch("symbiont.llms.ChatOpenAI")
def test_init_llm_openai_model(mock_chat_openai, mock_is_openai_model):
    """
    Tests the initialization of an OpenAI language model.

    Verifies that the `init_llm` function correctly initializes an OpenAI model
    when given a valid API key and model name.

    Args:
        mock_chat_openai (Mock): A mock object for the `ChatOpenAI` class.
        mock_is_openai_model (Mock): A mock object for the `isOpenAImodel` function.

    Returns:
        None
    """
    settings = create_settings("gpt-3")
    api_key = "openai_api_key"
    init_llm(settings, api_key)
    mock_chat_openai.assert_called_once_with(model="gpt-3", api_key=api_key, max_tokens=1500, temperature=0.7)


@patch("symbiont.llms.isAnthropicModel", return_value=True)
@patch("symbiont.llms.ChatAnthropic")
def test_init_llm_anthropic_model(mock_chat_anthropic, mock_is_anthropic_model):
    settings = create_settings("claude")
    api_key = "anthropic_api_key"
    init_llm(settings, api_key)
    mock_chat_anthropic.assert_called_once_with(model_name="claude", api_key=api_key, temperature=0.7, timeout=60)


@patch("symbiont.llms.isGoogleModel", return_value=True)
@patch("symbiont.llms.ChatGoogleGenerativeAI")
def test_init_llm_google_model(mock_chat_google, mock_is_google_model):
    settings = create_settings("palm")
    api_key = "google_api_key"
    init_llm(settings, api_key)
    mock_chat_google.assert_called_once_with(
        model="palm",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True,
        client_options={"max_output_tokens": 1500},
        transport=None,
        client=None,
    )


@patch("symbiont.llms.isOpenAImodel", return_value=False)
@patch("symbiont.llms.isAnthropicModel", return_value=False)
@patch("symbiont.llms.isGoogleModel", return_value=False)
def test_init_llm_unsupported_model(mock_is_openai_model, mock_is_anthropic_model, mock_is_google_model):
    """
    Test the initialization of an unsupported language model by mocking the model checks.

    This function tests the behavior of the `init_llm` function when the language model being
    initialized is not supported. It mocks the `isOpenAImodel`, `isAnthropicModel`, and
    `isGoogleModel` functions to return `False` for all three checks. It then calls the
    `init_llm` function with a settings object created using the `create_settings` function
    and an API key of "some_api_key". The function expects an `HTTPException` to be raised
    with a status code of 400 and a detail message of "Error initializing LLM".

    Parameters:
        None

    Returns:
        None
    """
    settings = create_settings("unsupported_model")
    api_key = "some_api_key"
    with pytest.raises(HTTPException) as exc_info:
        init_llm(settings, api_key)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Error initializing LLM"


def test_init_llm_validation_error():
    """
    Test the initialization of the language model with validation error.

    This test function checks if the `init_llm` function raises a `ValidationError` when
    the `llm_name` property of the `UsersLLMSettings` object is set to `None`. It does this
    by calling the `init_llm` function with a `UsersLLMSettings` object created with `llm_name`
    set to `None` and an `api_key` of `"api_key"`. The test asserts that a `ValidationError`
    is raised.

    Parameters:
        None

    Returns:
        None
    """
    with pytest.raises(ValidationError):
        init_llm(UsersLLMSettings(llm_name=None), api_key="api_key")


def test_isOpenAImodel():
    assert isOpenAImodel("gpt-3.5-turbo")
    assert isOpenAImodel("gpt-4")
    assert not isOpenAImodel("not-gpt")
    assert isOpenAImodel("gpt")


def test_isAnthropicModel():
    assert isAnthropicModel("claude-v1")
    assert isAnthropicModel("claude-v2")
    assert not isAnthropicModel("not-claude")
    assert isAnthropicModel("claude")


def test_isGoogleModel():
    assert isGoogleModel("models/gemini-1")
    assert isGoogleModel("gemini-2")
    assert not isGoogleModel("not-gemini")
    assert isGoogleModel("models/gemini")


if __name__ == "__main__":
    pytest.main()
