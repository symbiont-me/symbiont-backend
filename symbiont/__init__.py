import os
from .utils.logger import Logger
from dotenv import load_dotenv, find_dotenv

VERSION = "0.2.0"

logger = Logger(__name__, environment=os.getenv("FASTAPI_ENV"))
ENVIRONMENT = os.getenv("FASTAPI_ENV", default="development")

if ENVIRONMENT == "production":
    load_dotenv(find_dotenv(".env.production"))
    logger.critical(f"Running in production environment with version: {VERSION}")
if ENVIRONMENT == "development":
    load_dotenv(find_dotenv(".env.development"))
    logger.info(f"Running in development environment with version: {VERSION}")
