import pytest
from symbiont.llms import init_llm, UsersLLMSettings
from unittest.mock import patch


# Fixtures
@pytest.fixture
def mock_chat_openai():
    with patch("symbiont.llms.ChatOpenAI") as mock:
        yield mock


@pytest.fixture
def mock_chat_anthropic():
    with patch("symbiont.llms.ChatAnthropic") as mock:
        yield mock


@pytest.fixture
def mock_chat_google():
    with patch("symbiont.llms.ChatGoogleGenerativeAI") as mock:
        yield mock


# Helper function to ensure default values are set on the UsersLLMSettings
def assert_common_settings(settings):
    assert settings.max_tokens == 1500
    assert settings.temperature == 0.7
    assert settings.timeout == 60


# Tests for different LLM services
@patch("symbiont.llms.ChatOpenAI")
def test_openai_is_initialized(mock_chat_openai):
    settings = UsersLLMSettings(llm_name="gpt-3.5-turbo")
    assert_common_settings(settings)
    api_key = "test_api_key"
    llm = init_llm(settings, api_key)
    mock_chat_openai.assert_called_once_with(
        model=settings.llm_name, api_key=api_key, max_tokens=settings.max_tokens, temperature=settings.temperature
    )
    assert llm is mock_chat_openai.return_value


@patch("symbiont.llms.ChatAnthropic")
def test_anthropic_is_initialized(mock_chat_anthropic):
    settings = UsersLLMSettings(llm_name="claude-v1")
    assert_common_settings(settings)
    api_key = "test_api_key"
    llm = init_llm(settings, api_key)
    mock_chat_anthropic.assert_called_once_with(
        model_name=settings.llm_name, api_key=api_key, temperature=settings.temperature, timeout=settings.timeout
    )
    assert llm is mock_chat_anthropic.return_value


@patch("symbiont.llms.ChatGoogleGenerativeAI")
def test_google_is_initialized(mock_chat_google):
    settings = UsersLLMSettings(llm_name="gemini-1.0-pro")
    assert_common_settings(settings)
    api_key = "test_api_key"
    llm = init_llm(settings, api_key)
    mock_chat_google.assert_called_once_with(
        model=settings.llm_name,
        google_api_key=api_key,
        temperature=settings.temperature,
        convert_system_message_to_human=True,
        client_options={"max_output_tokens": settings.max_tokens},  # @note not sure if this is working
        transport=None,
        client=None,
    )
    assert llm is mock_chat_google.return_value


def test_no_llm_is_initialized():
    settings = UsersLLMSettings(llm_name="apple")
    assert_common_settings(settings)
    api_key = "test_api_key"
    llm = init_llm(settings, api_key)
    assert llm is None
