import requests
import os


async def download_from_firebase_storage(file_key: str, download_url: str):
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        save_path = f"temp/{file_key}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully and saved to {save_path}")
        return save_path
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


# TODO maybe delete the entire directory because we do not want to keep the user id in the file path
async def delete_local_file(file_path: str):
    os.remove(file_path)
    print(f"File deleted successfully from {file_path}")
