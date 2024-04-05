import os
from .utils.logger import Logger

logger = Logger(__name__, environment=os.getenv("FASTAPI_ENV"))
if os.getenv("FASTAPI_ENV") is None:
    logger.info("Run: 'export FASTAPI_ENV=development' in terminal")
    raise ValueError("FASTAPI_ENV environment variable not set")
if os.getenv("FASTAPI_ENV") == "production":
    logger.warning("Running in production environment")
else:
    logger.warning(f"Environment: {os.getenv('FASTAPI_ENV')}")
