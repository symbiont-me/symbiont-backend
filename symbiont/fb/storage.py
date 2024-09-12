import requests
import os
from .. import logger


# TODO maybe delete the entire directory because
# we do not want to keep the user id in the file path
async def delete_local_file(local_file_path: str):
    os.remove(local_file_path)
    logger.info(f"Deleted file {local_file_path}")
