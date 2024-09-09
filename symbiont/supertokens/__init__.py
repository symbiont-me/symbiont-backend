import os
from supertokens_python import init, InputAppInfo, SupertokensConfig
from supertokens_python.recipe import session, emailpassword

import re


def init_supertokens():
    """
    Initializes the Supertokens configuration for the application.

    Checks the environment variable "AUTH" to determine whether to use production or development settings.
    If "AUTH" is set to "production", it expects the environment variables "PROD_API_DOMAIN", "PROD_WEBSITE_DOMAIN",
    and "PROD_CONNECTION_URI" to be set.
    If "AUTH" is not set to "production", it uses default development settings.

    Initializes the Supertokens configuration with the specified app info, connection URI, framework, and recipe list.
    """
    auth_env = os.getenv("AUTH")

    if auth_env == "production":
        api_domain = os.getenv("PROD_API_DOMAIN")
        website_domain = os.getenv("PROD_WEBSITE_DOMAIN")
        connection_uri = os.getenv("PROD_CONNECTION_URI")

        if not api_domain or not website_domain or not connection_uri:
            raise ValueError(
                "Production environment variables PROD_API_DOMAIN, PROD_WEBSITE_DOMAIN, and PROD_CONNECTION_URI must be set"
            )
    else:
        api_domain = "http://127.0.0.1:8000"

        website_domain = "http://localhost:3000"
        if not re.match(r"^http://localhost:(300[0-9]|3010)$", website_domain):
            raise ValueError("Port number must be between 3001 and 3010")
        connection_uri = "localhost:3567"

    init(
        app_info=InputAppInfo(
            app_name="symbiont",
            api_domain=api_domain,
            website_domain=website_domain,
            api_base_path="/auth",
            website_base_path="/auth",
        ),
        supertokens_config=SupertokensConfig(
            connection_uri=connection_uri,
        ),
        framework="fastapi",
        recipe_list=[
            session.init(
                expose_access_token_to_frontend_in_cookie_based_auth=True,
                cookie_secure=True,
            ),
            emailpassword.init(),
        ],
        mode="wsgi",
    )
