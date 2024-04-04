from ..fb.storage import delete_local_file
import os
import pytest


async def test_delete_local_file():
    new_dir = "temp"
    os.makedirs(new_dir, exist_ok=True)
    file_path = "temp/temp.txt"
    with open(file_path, "w") as f:
        f.write("Mistah Kurtz - He Dead!")
    assert os.path.exists(file_path)
    await delete_local_file(file_path)
    assert not os.path.exists(file_path)


async def test_delete_non_existent_file():
    non_existent_file_path = "temp/non_existent_file.txt"
    with pytest.raises(FileNotFoundError):
        await delete_local_file(non_existent_file_path)
