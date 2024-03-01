# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       HELPER FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
from typing import Optional
from fastapi import HTTPException, Header
from firebase_admin import auth


def remove_non_ascii(text):
    return "".join(i for i in text if ord(i) < 128)


def replace_space_with_underscore(text):
    return text.replace(" ", "_")


def make_file_identifier(text):
    cleaned_filename = remove_non_ascii(replace_space_with_underscore(text))
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    identifier = f"{current_datetime}_{cleaned_filename}"
    return identifier


async def verify_user_auth_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(
            status_code=401, detail="Authorization header missing")
    try:
        id_token = authorization.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(id_token)

        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
