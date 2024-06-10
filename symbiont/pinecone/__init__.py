from dotenv import load_dotenv
import os
from pinecone import Pinecone
import time
from .. import logger
import time


load_dotenv()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       PINECONE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
pinecone_endpoint = os.getenv("PINECONE_API_ENDPOINT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")


if pinecone_index_name is None:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set")
start_time = time.time()
pc = Pinecone(api_key=pinecone_api_key, endpoint=pinecone_endpoint)
# wait for index to be initialized
while not pc.describe_index(pinecone_index_name).status["ready"]:
    time.sleep(1)

pc_index = pc.Index(pinecone_index_name)

if pc_index is None:
    raise Exception("Pinecone index not found")
elapsed_time = time.time() - start_time
logger.info(f"Pinecone initialized: {pinecone_index_name} in {elapsed_time}")
