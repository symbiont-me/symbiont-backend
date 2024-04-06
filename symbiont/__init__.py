import os
from .utils.logger import Logger
from dotenv import load_dotenv


logger = Logger(__name__, environment=os.getenv("FASTAPI_ENV"))
ENVIRONMENT = os.getenv("FASTAPI_ENV", default="development")

if ENVIRONMENT == "production":
    load_dotenv(".env.production")
    logger.critical("Running in production environment")
if ENVIRONMENT == "development":
    load_dotenv(".env.development")
    logger.info("Running in development environment")



