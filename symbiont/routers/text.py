from fastapi import APIRouter, Request, HTTPException
from ..models import TextUpdateRequest
from ..mongodb import studies_collection
from symbiont.mongodb.utils import user_exists, check_user_authorization

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       STUDY TEXT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

router = APIRouter()


@router.post("/update-text")
async def update_text(text_request: TextUpdateRequest, request: Request):
    session_data = {
        "user_id": request.state.session.get_user_id(),
    }

    user_uid = session_data["user_id"]
    await user_exists(user_uid)
    check_user_authorization(text_request.studyId, user_uid, studies_collection)
    studies_collection.update_one(
        {"_id": text_request.studyId}, {"$set": {"text": text_request.text}}
    )
    return {"message": "Text updated successfully"}
