import os
from .utils.logger import Logger

if os.getenv("FASTAPI_ENV") is None:
    logger.info("Run: 'export FASTAPI_ENV=development' in terminal")
    raise ValueError("FASTAPI_ENV environment variable not set")
logger = Logger(__name__, environment=os.getenv("FASTAPI_ENV"))
logger.info(f"Logger initialized\tEnvironment: {os.getenv('FASTAPI_ENV')}")
