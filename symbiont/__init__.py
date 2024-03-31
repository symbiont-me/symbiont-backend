import os
from .utils.logger import Logger

logger = Logger(__name__, environment=os.getenv("FASTAPI_ENV"))
logger.info(f"Logger initialized\tEnvironment: {os.getenv('FASTAPI_ENV')}")
