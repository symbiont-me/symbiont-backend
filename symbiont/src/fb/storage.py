import requests
import os
from .. import logger


async def download_from_firebase_storage(resource_identifier: str, download_url: str):
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        save_path = f"temp/{resource_identifier}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"File downloaded successfully to {save_path}")
        return save_path
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP error occurred: {err}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    return None


# TODO maybe delete the entire directory because we do not want to keep the user id in the file path
async def delete_local_file(local_file_path: str):
    os.remove(local_file_path)
    logger.info(f"Deleted file {local_file_path}")
