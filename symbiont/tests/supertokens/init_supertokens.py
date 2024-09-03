import pytest
from unittest.mock import patch
from symbiont.supertokens import init_supertokens


@patch("symbiont.supertokens.init")
@patch("os.getenv")
def test_init_supertokens_development(mock_getenv, mock_init):
    # Test with default (development) values
    mock_getenv.side_effect = lambda key: None if key == "AUTH" else "default_value"
    init_supertokens()

    mock_init.assert_called_once()
    args, kwargs = mock_init.call_args

    assert kwargs["app_info"].app_name == "symbiont"
    assert kwargs["app_info"].api_domain == "http://127.0.0.1:8000"
    assert kwargs["app_info"].website_domain == "http://localhost:3000"
    assert kwargs["app_info"].api_base_path == "/auth"
    assert kwargs["app_info"].website_base_path == "/auth"
    assert kwargs["supertokens_config"].connection_uri == "localhost:3567"
    assert kwargs["framework"] == "fastapi"
    assert kwargs["mode"] == "wsgi"
    assert len(kwargs["recipe_list"]) == 2


@patch("symbiont.supertokens.init")
@patch("os.getenv")
def test_init_supertokens_production(mock_getenv, mock_init):
    # Test with production values
    mock_getenv.side_effect = lambda key: {
        "AUTH": "production",
        "PROD_API_DOMAIN": "https://api.production.com",
        "PROD_WEBSITE_DOMAIN": "https://www.production.com",
        "PROD_CONNECTION_URI": "prod-db:3567",
    }.get(key, None)
    init_supertokens()

    mock_init.assert_called_once()
    args, kwargs = mock_init.call_args

    assert kwargs["app_info"].app_name == "symbiont"
    assert kwargs["app_info"].api_domain == "https://api.production.com"
    assert kwargs["app_info"].website_domain == "https://www.production.com"
    assert kwargs["app_info"].api_base_path == "/auth"
    assert kwargs["app_info"].website_base_path == "/auth"
    assert kwargs["supertokens_config"].connection_uri == "prod-db:3567"
    assert kwargs["framework"] == "fastapi"
    assert kwargs["mode"] == "wsgi"
    assert len(kwargs["recipe_list"]) == 2


@patch("symbiont.supertokens.init")
@patch("os.getenv")
def test_init_supertokens_production_missing_values(mock_getenv, mock_init):
    # Test with missing production values
    mock_getenv.side_effect = lambda key: "production" if key == "AUTH" else None
    with pytest.raises(
        ValueError,
        match="Production environment variables PROD_API_DOMAIN, PROD_WEBSITE_DOMAIN, and PROD_CONNECTION_URI must be set",
    ):
        init_supertokens()
