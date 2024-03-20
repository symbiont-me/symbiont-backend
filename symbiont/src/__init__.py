import os
from .utils.logger import Logger

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       OPENAI INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
openai_api_key = os.getenv("OPENAI_API_KEY")
logger_instance = Logger(__name__, environment=os.getenv("FASTAPI_ENV"))
